import os
import gradio as gr
import requests
import inspect
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from crewai_tools import FileReadTool
from duckduckgo_search import DDGS
from crewai.tools import BaseTool

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

class WebSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = "Searches the web for information using DuckDuckGo."

    def _run(self, query: str) -> str:
        print(f"--- TOOL: Melakukan pencarian web untuk '{query}' ---")
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=5)]
                return str(results) if results else "Tidak ada hasil ditemukan."
        except Exception as e:
            return f"Error saat melakukan pencarian: {e}"

class CrewAIAgent:
    def __init__(self):
        # Solusi 1: Ubah nama model ke format yang benar untuk Google Gemini
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # Ubah dari "gemini-1.5-flash-latest"
            verbose=True,
            temperature=0.5,
            max_retries=2,
            request_timeout=30,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        # Alternatif solusi 2: Jika masih error, coba model name yang lebih spesifik
        # self.llm = ChatGoogleGenerativeAI(
        #     model="models/gemini-1.5-flash",
        #     verbose=True,
        #     temperature=0.5,
        #     google_api_key=os.getenv("GEMINI_API_KEY")
        # )

        # Inisialisasi tools
        web_search_tool = WebSearchTool()
        file_read_tool = FileReadTool()

        # Buat agent
        self.gaia_solver = Agent(
            role='GAIA Problem Solver',
            goal='Accurately answer complex questions based on provided context and by using available tools.',
            backstory=(
                "You are an expert AI assistant. You are methodical, you break down problems into steps, "
                "and you use your available tools like web search or file reading to find evidence. "
                "After you have gathered enough information, you provide the final answer directly."
            ),
            verbose=True,
            allow_delegation=False,
            tools=[web_search_tool, file_read_tool],
            llm=self.llm
        )

        self.crew = Crew(
            agents=[self.gaia_solver],
            tasks=[],
            process=Process.sequential,
            verbose=True
        )

    def __call__(self, question: str) -> str:
        solve_task = Task(
            description=f"Solve the following problem: {question}",
            expected_output="The final, concise answer to the problem.",
            agent=self.gaia_solver
        )

        self.crew.tasks = [solve_task]
        try:
            result = self.crew.kickoff()
            return result
        except Exception as e:
            print(f"---!! An error occurred during crew kickoff: {e} !!---")
            return f"AGENT FAILED: {e}"


# Alternative CrewAI Agent dengan konfigurasi berbeda
class CrewAIAgentAlternative:
    def __init__(self):
        # Solusi untuk tier gratis: gunakan model dengan rate limit terbaik
        model_name = os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-flash")  # Default ke Flash

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            verbose=True,
            temperature=0.5,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            # Optimasi untuk tier gratis
            convert_system_message_to_human=True,
            max_retries=2,
            request_timeout=30
        )

        # Inisialisasi tools
        web_search_tool = WebSearchTool()
        file_read_tool = FileReadTool()

        # Buat agent dengan error handling yang lebih baik
        self.gaia_solver = Agent(
            role='GAIA Problem Solver',
            goal='Accurately answer complex questions based on provided context and by using available tools.',
            backstory=(
                "You are an expert AI assistant. You are methodical, you break down problems into steps, "
                "and you use your available tools like web search or file reading to find evidence. "
                "After you have gathered enough information, you provide the final answer directly."
            ),
            verbose=True,
            allow_delegation=False,
            tools=[web_search_tool, file_read_tool],
            llm=self.llm
        )

        self.crew = Crew(
            agents=[self.gaia_solver],
            tasks=[],
            process=Process.sequential,
            verbose=True
        )

    def __call__(self, question: str) -> str:
        solve_task = Task(
            description=f"Solve the following problem: {question}",
            expected_output="The final, concise answer to the problem.",
            agent=self.gaia_solver
        )

        self.crew.tasks = [solve_task]
        try:
            # Test LLM connection terlebih dahulu
            test_response = self.llm.invoke("Test connection")
            print(f"LLM Test successful: {test_response}")

            result = self.crew.kickoff()
            return str(result)
        except Exception as e:
            print(f"---!! An error occurred during crew kickoff: {e} !!---")
            return f"AGENT FAILED: {e}"


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent dengan error handling yang lebih baik
    try:
        # Coba agent utama terlebih dahulu
        agent = CrewAIAgent()
        # Test agent dengan pertanyaan sederhana
        test_result = agent("What is 2+2?")
        if "AGENT FAILED" in test_result:
            print("Primary agent failed, trying alternative...")
            agent = CrewAIAgentAlternative()
    except Exception as e:
        print(f"Error instantiating primary agent: {e}")
        try:
            print("Trying alternative agent configuration...")
            agent = CrewAIAgentAlternative()
        except Exception as e2:
            print(f"Error instantiating alternative agent: {e2}")
            return f"Error initializing both agent configurations: Primary: {e}, Alternative: {e2}", None

    # Rest of the function remains the same...
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON response from questions endpoint: {e}")
        print(f"Response text: {response.text[:500]}")
        return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)
