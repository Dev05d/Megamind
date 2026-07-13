import questionary
from core.retriever import ask_megamind
from scrapers.scrape_pdf     import ingest_pdf
from scrapers.scrape_web     import ingest_web
from scrapers.scrape_youtube import ingest_youtube
from scrapers.scrape_reddit  import ingest_reddit
from core.tracker import log_ingestion, backup_pdf
from tools.show_raw_data import show_raw_data
from tools.delete_source import delete_source 
from tools.show_db_info import show_db_info
from tools.change_embedd_model import change_embedd_model
from tools.model_migrator import change_chat_model, change_router_model


def interactive_menu():
    """Displays the interactive arrow-key menu."""
    while True:
        action = questionary.select(
            "🧠 What would you like Megamind to do?",
            choices=[
                "Ask a Question",
                "Ingest Web Article",
                "Ingest PDF",
                "Ingest YouTube Video",
                "Ingest Reddit Post",
                "Tools...",  
                "Exit"
            ]
        ).ask()

        #  Main Menu Actions 
        if action == "Ask a Question":
            query = questionary.text("Enter your question:").ask()
            if query: ask_megamind(query)
        

        elif action == "Ingest Web Article":
            url = questionary.text("Enter URL:").ask()
            if url: 
                ingest_web(url)
                log_ingestion("web", url)

        elif action == "Ingest PDF":
            path = questionary.path("Enter PDF file path:").ask()
            if path:
                clean_path = path.strip().replace("\\ ", " ").strip("'\"")
                ingest_pdf(clean_path)
                backup_pdf(clean_path)

        elif action == "Ingest YouTube Video":
            url = questionary.text("Enter YouTube URL:").ask()
            if url: 
                ingest_youtube(url)
                log_ingestion("youtube", url)

        elif action == "Ingest Reddit Post":
            url = questionary.text("Enter Reddit Post URL:").ask()
            if url: 
                ingest_reddit(url)
                log_ingestion("reddit", url)

        # Nested Tools Sub-Menu 
        elif action == "Tools...":
            tool_action = questionary.select(
                "🛠️ Select a Tool:",
                choices=[
                    "Change Chat Model & Context Window",
                    "Change Router Model",
                    "Database Info",
                    "Show Raw Chunk Data",
                    "Delete a Source",
                    "Change Embedding",
                    "Back to Main Menu"
                ]
            ).ask()

            if tool_action == "Change Chat Model & Context Window":
                change_chat_model()
            if tool_action == "Change Router Model":
                change_router_model()
            elif tool_action == "Database Info":
                show_db_info()
            elif tool_action == "Show Raw Chunk Data":
                show_raw_data()
            elif tool_action == "Delete a Source":
                delete_source()
            elif tool_action == "Change Embedding":
                change_embedd_model()
            elif tool_action in ("Back to Main Menu", None):
                continue  # Loops back to the main menu choice

        #  Exit App 
        elif action in ("Exit", None):
            print("Shutting down Megamind. Goodbye!")
            break