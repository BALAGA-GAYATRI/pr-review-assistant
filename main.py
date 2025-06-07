import os
import dotenv
import requests
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

# Load environment variables
dotenv.load_dotenv()

# Define plugin class with detailed prompts
class PRReviewPlugin:
    def summarize(self, input: str) -> str:
        return f"""
        You are an expert software engineer. Summarize the changes in the GitHub pull request diff below.

        Focus on:
        - What files were changed
        - What kind of changes were made (e.g., bug fixes, new features, refactors)
        - The overall purpose of the changes

        Be clear and concise. Avoid excessive technical jargon.

        Diff:
        {input}

        Summary:
        """

    def critique(self, input: str) -> str:
        return f"""
        You are a senior software engineer reviewing the following GitHub pull request diff.

        Provide constructive feedback on:
        - Possible bugs or logic issues
        - Code style or best practice concerns
        - Opportunities to improve readability, performance, or structure
        - Missing documentation or tests

        Be professional and helpful.

        Diff:
        {input}

        Critique:
        """

    def generate_comment(self, input: str) -> str:
        return f"""
        You are a friendly and professional GitHub bot. Use the summary and critique below to generate a constructive comment for a pull request.

        Make sure to:
        - Start with a positive note
        - Present the summary clearly
        - Highlight important points from the critique
        - Encourage collaboration and improvement

        Input:
        {input}

        GitHub PR Comment:
        """

# Initialize Semantic Kernel and Azure OpenAI chat service
kernel = Kernel()
azure_openai_chat = AzureChatCompletion(
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
kernel.add_service(azure_openai_chat, "chat")

# Register the plugin
plugin = PRReviewPlugin()
kernel.add_plugin(plugin, "pr_review_plugin")

# GitHub API helpers
def get_pr_diff():
    headers = {
        "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
        "Accept": "application/vnd.github.v3.diff"
    }
    repo = os.getenv("GITHUB_REPO")
    pr_number = os.getenv("GITHUB_PR_NUMBER")
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    response = requests.get(url, headers=headers)
    if response.ok:
        return response.text
    print(f"❌ Failed to fetch PR diff: {response.status_code} {response.text}")
    return None

def post_comment(comment: str):
    headers = {
        "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
        "Accept": "application/vnd.github+json"
    }
    repo = os.getenv("GITHUB_REPO")
    pr_number = os.getenv("GITHUB_PR_NUMBER")
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    response = requests.post(url, headers=headers, json={"body": comment})
    if not response.ok:
        print(f"❌ Failed to post comment: {response.status_code} {response.text}")

# Main review pipeline
async def run_review():
    print("📥 Fetching PR diff...")
    diff = get_pr_diff()
    if not diff:
        return

    print("✅ Diff fetched. Generating summary...")
    summary_prompt = plugin.summarize(diff)
    summary_result = await kernel.invoke_prompt(summary_prompt)
    print(summary_result)

    print("✅ Summary complete. Generating critique...")
    critique_prompt = plugin.critique(diff)
    critique_result = await kernel.invoke_prompt(critique_prompt)
    print(critique_result)
    combined_input = f"Summary:\n{summary_result}\n\nCritique:\n{critique_result}"

    print("✅ Generating GitHub comment...")
    comment_prompt = plugin.generate_comment(combined_input)
    comment_result = await kernel.invoke_prompt(comment_prompt)
    print(comment_result)

    print("✅ Posting comment...")
    post_comment(str(comment_result))
    print("✅ Done.")

if __name__ == "__main__":
    asyncio.run(run_review())
