# HAPUS ATAU KOMENTARI CLASS MyLLMAgent YANG LAMA

# --- IMPLEMENTASI BARU MENGGUNAKAN CREWAI ---
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool, FileReadTool

# --- 1. DEFINISIKAN ALAT (TOOLS) ---
# Inisialisasi alat pencarian web. Kamu perlu API key dari serper.dev untuk ini,
# atau kita bisa gunakan duckduckgo seperti sebelumnya. Mari kita mulai dengan yang ini.
# Jangan lupa tambahkan SERPER_API_KEY di environment variables/HF Secrets
# web_search_tool = SerperDevTool()

# Alternatif jika tidak punya API Key Serper, kita bisa definisikan tool sendiri:
from duckduckgo_search import DDGS
from crewai.tools import BaseTool

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

web_search_tool = ScrapeWebsiteTool()
file_read_tool = FileReadTool()


# --- 2. BUAT CLASS AGEN BARU UNTUK INTEGRASI ---
class CrewAIAgent:
    def __init__(self):
        # Setup LLM. CrewAI butuh ini untuk tahu "otak" mana yang harus dipakai.
        # Pastikan GOOGLE_API_KEY sudah di-set sebagai environment variable.
        os.environ["OPENAI_API_BASE"] = 'https://openrouter.ai/api/v1' # Ini untuk LiteLLM
        os.environ["OPENAI_MODEL_NAME"] = 'google/gemini-flash-1.5' # Ganti dengan modelmu
        os.environ["OPENAI_API_KEY"] = os.getenv("GOOGLE_API_KEY") # Atau API key-mu yg lain


    def __call__(self, question: str) -> str:
        # --- 3. DEFINISIKAN AGEN DAN TUGAS DI DALAM SETIAP PANGGILAN ---

        # Definisikan "pegawai"-mu. Seorang problem solver serba bisa.
        gaia_solver = Agent(
            role='GAIA Problem Solver',
            goal='Accurately answer complex questions based on provided context and by using available tools.',
            backstory=(
                "You are an expert AI assistant. You are methodical, you break down problems into steps, "
                "and you use your available tools (like web search or file reading) to find evidence. "
                "After you have gathered enough information, you provide the final answer directly."
            ),
            verbose=True,
            allow_delegation=False,
            tools=[web_search_tool, file_read_tool]  # <--- MEMBERIKAN ALAT KE AGEN
        )

        # Definisikan tugas yang harus dikerjakan oleh agen.
        solve_task = Task(
            description=f"Solve the following problem: {question}",
            expected_output="The final, concise answer to the problem.",
            agent=gaia_solver
        )

        # --- 4. BENTUK TIM (CREW) DAN MULAI BEKERJA ---
        crew = Crew(
            agents=[gaia_solver],
            tasks=[solve_task],
            process=Process.sequential,
            verbose=2 # Ini akan mencetak semua proses pikir agen, sangat bagus untuk belajar!
        )

        # Jalankan tugasnya dan dapatkan hasilnya
        result = crew.kickoff()
        return result
