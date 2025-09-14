import os
import gradio as gr
import requests
import inspect
import pandas as pd
import time
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
        # PERBAIKAN UTAMA: Format model name yang benar untuk Google Gemini
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # TANPA "models/" prefix
            verbose=True,
            temperature=0.5,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            max_retries=2,
            request_timeout=30
        )

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
            # Tambah delay untuk rate limiting (15 RPM = 4 detik per request)
            time.sleep(4)

            result = self.crew.kickoff()
            return str(result)
        except Exception as e:
            print(f"---!! An error occurred during crew kickoff: {e} !!---")
            return f"AGENT FAILED: {e}"


# Alternatif dengan konfigurasi lebih eksplisit
class CrewAIAgentAlternative:
    def __init__(self):
        # Konfigurasi yang lebih eksplisit untuk menghindari masalah litellm
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        # Format model yang benar sesuai dokumentasi litellm untuk Google
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # Format standar tanpa prefix
            google_api_key=api_key,
            temperature=0.5,
            verbose=True,
            max_tokens=1000,  # Tambah limit token untuk kontrol
            max_retries=2,
            request_timeout=30,
            # Jangan gunakan parameter deprecated
        )

        # Tools
        web_search_tool = WebSearchTool()
        file_read_tool = FileReadTool()

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

    def test_llm_connection(self):
        """Test LLM connection sebelum digunakan"""
        try:
            response = self.llm.invoke("Say hello")
            print(f"✅ LLM Test berhasil: {response.content}")
            return True
        except Exception as e:
            print(f"❌ LLM Test gagal: {e}")
            return False

    def __call__(self, question: str) -> str:
        # Test connection dulu
        if not self.test_llm_connection():
            return "AGENT FAILED: LLM connection test failed"

        solve_task = Task(
            description=f"Solve the following problem: {question}",
            expected_output="The final, concise answer to the problem.",
            agent=self.gaia_solver
        )

        self.crew.tasks = [solve_task]
        try:
            # Rate limiting untuk tier gratis (15 RPM)
            time.sleep(4)

            result = self.crew.kickoff()
            return str(result)
        except Exception as e:
            print(f"---!! An error occurred during crew kickoff: {e} !!---")
            return f"AGENT FAILED: {e}"


# Solusi fallback menggunakan OpenAI format jika Gemini masih bermasalah
class CrewAIAgentOpenAI:
    def __init__(self):
        try:
            from langchain_openai import ChatOpenAI

            # Jika ada OpenAI API key, gunakan itu
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                self.llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    api_key=openai_key,
                    temperature=0.5,
                    max_tokens=1000
                )
            else:
                raise ValueError("No OpenAI API key provided")

        except ImportError:
            raise ValueError("langchain_openai not installed")

        # Tools
        web_search_tool = WebSearchTool()
        file_read_tool = FileReadTool()

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

    # 1. Instantiate Agent dengan multiple fallback
    agent = None
    agent_type = "Unknown"

    # Try 1: Primary Gemini Agent
    try:
        print("🔄 Trying primary Gemini agent...")
        agent = CrewAIAgent()
        test_result = agent("What is 2+2?")
        if "AGENT FAILED" not in test_result:
            agent_type = "Primary Gemini"
            print("✅ Primary Gemini agent working")
        else:
            agent = None
    except Exception as e:
        print(f"❌ Primary agent failed: {e}")

    # Try 2: Alternative Gemini Agent
    if agent is None:
        try:
            print("🔄 Trying alternative Gemini agent...")
            agent = CrewAIAgentAlternative()
            agent_type = "Alternative Gemini"
            print("✅ Alternative Gemini agent working")
        except Exception as e:
            print(f"❌ Alternative Gemini agent failed: {e}")

    # Try 3: OpenAI Fallback (jika tersedia)
    if agent is None and os.getenv("OPENAI_API_KEY"):
        try:
            print("🔄 Trying OpenAI fallback agent...")
            agent = CrewAIAgentOpenAI()
            agent_type = "OpenAI Fallback"
            print("✅ OpenAI fallback agent working")
        except Exception as e:
            print(f"❌ OpenAI fallback agent failed: {e}")

    if agent is None:
        return """
        🚫 All agent configurations failed. Please check:
        1. GEMINI_API_KEY environment variable is set correctly
        2. API key has proper permissions
        3. Try adding OPENAI_API_KEY as fallback
        4. Check network connectivity
        """, None

    print(f"🎯 Using agent: {agent_type}")

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
        return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running {agent_type} agent on {len(questions_data)} questions...")

    for i, item in enumerate(questions_data):
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue

        print(f"🔄 Processing question {i+1}/{len(questions_data)}: {task_id}")

        try:
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
            print(f"✅ Question {i+1} completed")
        except Exception as e:
            print(f"❌ Error running agent on task {task_id}: {e}")
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"{agent_type} agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"Agent Used: {agent_type}\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except Exception as e:
        status_message = f"Submission Failed: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# CrewAI Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1. **Set Environment Variables** (di Hugging Face Space Settings):
           - `GEMINI_API_KEY`: Your Google AI Studio API key (free)
           - `OPENAI_API_KEY`: Optional OpenAI key as fallback

        2. **Login**: Use the button below to login with your Hugging Face account

        3. **Run**: Click 'Run Evaluation & Submit All Answers'

        **Model Info:**
        - Primary: Google Gemini 1.5 Flash (Free tier: 15 RPM)
        - Fallback: OpenAI GPT-3.5-turbo (if API key provided)

        ---
        **Note:** Processing takes time due to rate limiting (4 seconds per question for free tier).
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers", variant="primary")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=7, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 CrewAI Agent with Google Gemini")
    print("="*50)

    # Check environment variables
    if os.getenv("GEMINI_API_KEY"):
        print("✅ GEMINI_API_KEY found")
    else:
        print("❌ GEMINI_API_KEY not found - please set this environment variable")

    if os.getenv("OPENAI_API_KEY"):
        print("✅ OPENAI_API_KEY found (fallback available)")
    else:
        print("ℹ️  OPENAI_API_KEY not found (no fallback)")

    space_id = os.getenv("SPACE_ID")
    if space_id:
        print(f"✅ SPACE_ID: {space_id}")
    else:
        print("ℹ️  Running locally")

    print("="*50 + "\n")

    print("Launching Gradio Interface...")
    demo.launch(debug=True, share=False)
