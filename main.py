import os
import dotenv
import requests
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import OpenAIPromptExecutionSettings

# Load environment variables
dotenv.load_dotenv()

kernel = Kernel()
kernel.add_text_completion_service(
    "azure-openai-gpt",
    AzureChatCompletion(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
)
kernel.set_default_text_completion_service("azure-openai-gpt")

# Import skills from plugins
plugins = {}
for plugin_name in ["diff_reader", "change_summary", "review_critique", "comment_generator"]:
    plugins[plugin_name] = kernel.import_semantic_skill_from_directory("plugins", plugin_name)

# Get PR diff from GitHub
def get_pr_diff():
    headers = {"Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}", "Accept": "application/vnd.github.v3.diff"}
    repo = os.getenv("GITHUB_REPO")
    pr_number = os.getenv("GITHUB_PR_NUMBER")
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    response = requests.get(url, headers=headers)
    return response.text if response.ok else None

# Post a comment to PR
def post_comment(comment):
    headers = {"Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}", "Accept": "application/vnd.github+json"}
    repo = os.getenv("GITHUB_REPO")
    pr_number = os.getenv("GITHUB_PR_NUMBER")
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    requests.post(url, headers=headers, json={"body": comment})

import asyncio
async def run():
    diff = get_pr_diff()
    if not diff:
        print("❌ Failed to fetch PR diff")
        return

    print("✅ Diff fetched. Generating summary...")
    summary = await kernel.run_async(plugins["change_summary"]["skprompt"], input_str=diff)
    print("✅ Summary complete. Generating critique...")
    critique = await kernel.run_async(plugins["review_critique"]["skprompt"], input_str=diff)

    comment_prompt = f"""
    Pull Request Summary:
    {summary.result}

    Review Feedback:
    {critique.result}
    """

    print("✅ Generating GitHub comment...")
    comment = await kernel.run_async(plugins["comment_generator"]["skprompt"], input_str=comment_prompt)
    print("✅ Posting comment...")
    post_comment(comment.result)
    print("✅ Done.")

if __name__ == "__main__":
    asyncio.run(run())