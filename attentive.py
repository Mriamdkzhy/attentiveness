import boto3
import csv
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from langchain_aws import ChatBedrock
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics.g_eval import Rubric
from datasets import load_dataset

costs = []
input_prices = {
    "anthropic.claude-3-5-haiku-20241022-v1:0": 0.8,
    "us.meta.llama3-3-70b-instruct-v1:0": 0.72,
    "anthropic.claude-3-5-sonnet-20240620-v1:0": 3.00,
    "anthropic.claude-3-5-sonnet-20241022-v2:0": 3.00,
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": 3.00,
    "mistral.mistral-7b-instruct-v0:2": 0.15
}

output_prices = {
    "anthropic.claude-3-5-haiku-20241022-v1:0": 4,
    "us.meta.llama3-3-70b-instruct-v1:0": 0.72,
    "anthropic.claude-3-5-sonnet-20240620-v1:0": 15.00,
    "anthropic.claude-3-5-sonnet-20241022-v2:0": 15.00,
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": 15.00,
    "mistral.mistral-7b-instruct-v0:2": 0.20
}

# Define AWS Bedrock wrapper
class AWSBedrock(DeepEvalBaseLLM):
    def __init__(self, model_id, region="us-west-2"):
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)
        
    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
        )
        usage = response["usage"]
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)

        cost = (
            input_tokens * input_prices[self.model_id] +
            output_tokens * output_prices[self.model_id]
        ) / 1_000_000
        cost = round(cost, 6)
        costs.append(cost)
        return response["output"]["message"]["content"][0]["text"]

    async def a_generate(self, prompt: str) -> str:
        # DeepEval requires this even if you don't use async. You can fake it.
        from asyncio import to_thread
        return await to_thread(self.generate, prompt)

    def get_model_name(self) -> str:
        return self.model_id
    

# Load dataset
with open('data.csv', newline='') as csvfile:
    # reader = csv.reader(csvfile)
    reader = csv.DictReader(csvfile)
    data = list(reader)    

model_ids=["anthropic.claude-3-5-haiku-20241022-v1:0",
           "us.meta.llama3-3-70b-instruct-v1:0", 
           "anthropic.claude-3-5-sonnet-20240620-v1:0",
           "anthropic.claude-3-5-sonnet-20241022-v2:0",
           "us.anthropic.claude-3-7-sonnet-20250219-v1:0",  
           "mistral.mistral-7b-instruct-v0:2"]

model_names = {
    "anthropic.claude-3-5-haiku-20241022-v1:0": "Claude 3.5 Haiku",
    "us.meta.llama3-3-70b-instruct-v1:0": "Llama 3.3 70B",
    "anthropic.claude-3-5-sonnet-20240620-v1:0": "Claude 3.5 Sonnet v1",
    "anthropic.claude-3-5-sonnet-20241022-v2:0": "Claude 3.5 Sonnet v2",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": "Claude 3.7 Sonnet",
    "mistral.mistral-7b-instruct-v0:2": "Mistral 7B Instruct"
}

all_results = [["Model", "Test Case", "Attentiveness Score", "Attentiveness Reason"]]

for model_id in model_ids:
    aws_bedrock = AWSBedrock(model_id)

    # Define GEval metric
    attentive_metric = GEval(
        name="Attentiveness",
        criteria="Does the 'actual output' provide the short detail within 'expected output'?", # based on the 'input'
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5,
        model=aws_bedrock,
        rubric=[
            Rubric(score_range=(0, 0), expected_outcome="Detail not mentioned."),
            Rubric(score_range=(1, 5), expected_outcome="Some detail mentioned."),
            Rubric(score_range=(6, 9), expected_outcome="Most detail mentioned."),
            Rubric(score_range=(10, 10), expected_outcome="Detail clearly mentioned."),
        ]
    )

#  run with python3 <file_name>

test_results = [[f"Test Case", "Attentiveness Score", "Attentiveness Reason"]]

for model_id in model_ids:
    print(f"\nModel: {model_names[model_id]}")
    aws_bedrock = AWSBedrock(model_id)

    # Define GEval metric
    attentive_metric = GEval(
        name="Attentiveness",
        criteria="Does the 'actual output' provide the short detail within 'expected output'?",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5,
        model=aws_bedrock,
        rubric=[
            Rubric(score_range=(0, 0), expected_outcome="Detail not mentioned."),
            Rubric(score_range=(1, 5), expected_outcome="Some detail mentioned."),
            Rubric(score_range=(6, 9), expected_outcome="Most detail mentioned."),
            Rubric(score_range=(10, 10), expected_outcome="Detail clearly mentioned."),
        ]
    )

    # Loop through data
    for i in range(20):
        try:
            print(f"Test case {i+1}/20")
            row = data[i]
            summary = row['summary']
            detail = row['detail']
            prompt = "Summarise the following: " + summary
            actual = aws_bedrock.generate(prompt)

            test_case = LLMTestCase(
                name=f"Test Case {i+1}",
                input=summary,
                actual_output=actual,
                expected_output=detail
            )

            attentive_metric.measure(test_case)

            all_results.append([
                model_names[model_id],
                i+1,
                attentive_metric.score,
                attentive_metric.reason
            ])
        except Exception as e:
            print(f"Error in test case {i} for model {model_id}: {e}")

# Save all results to one CSV
output_filename = "attentive.csv"
with open(output_filename, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(all_results)