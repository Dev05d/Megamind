import argparse

from core.vector_store import init_db, close_db, _get_client
from core.retriever import ask_megamind
from scrapers.scrape_pdf     import ingest_pdf
from scrapers.scrape_web     import ingest_web
from scrapers.scrape_youtube import ingest_youtube
from scrapers.scrape_reddit  import ingest_reddit
from menutools import interactive_menu


def main() -> None:
    init_db()

    parser = argparse.ArgumentParser(
        description="Megamind — 100% Local Personal Knowledge Base",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--query",          type=str, metavar="QUESTION", help="Ask Megamind a question")
    parser.add_argument("--ingest-pdf",     type=str, metavar="PATH",     help="Ingest a PDF file")
    parser.add_argument("--ingest-web",     type=str, metavar="URL",      help="Ingest a web article")
    parser.add_argument("--ingest-youtube", type=str, metavar="URL",      help="Ingest a YouTube video")
    parser.add_argument("--ingest-reddit",  type=str, metavar="URL",      help="Ingest a Reddit post")

    args = parser.parse_args()

    # If any command line arguments are provided, use the old CLI behavior
    if any(vars(args).values()):
        if args.query:          ask_megamind(args.query)
        elif args.ingest_pdf:   ingest_pdf(args.ingest_pdf)
        elif args.ingest_web:   ingest_web(args.ingest_web)
        elif args.ingest_youtube: ingest_youtube(args.ingest_youtube)
        elif args.ingest_reddit:  ingest_reddit(args.ingest_reddit)
    
    # If NO arguments are provided, launch the interactive UI
    else:
        interactive_menu()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Megamind] Interrupted by user. Exiting...")
    except Exception as e:
        import traceback
        print("\n[Fatal Error Details]")
        traceback.print_exc()
    finally:
        close_db()