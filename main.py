import argparse
import sys
import yaml
from pathlib import Path
from src.processing.gather import InterviewDataCollector
from src.processing.index import InterviewIndexer
from src.processing.rag import InterviewRAGService
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml"):
    default_config = {
        "gather": {
            "data_dir": "data/raw",
            "max_results": 5,
            "include_sources": ["arxiv", "github", "web"]
        },
        "index": {
            "persist_dir": "data/processed",
            "chunk_size": 512,
            "chunk_overlap": 50
        },
        "rag": {
            "similarity_top_k": 4,
            "llm_model": "mistral-large-latest"
        }
    }

    if Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            for section in default_config:
                if section in user_config:
                    default_config[section].update(user_config[section])
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")

    return default_config


class InterviewPreparationPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.collector = InterviewDataCollector(self.config["gather"])
        self.indexer = InterviewIndexer(self.config["index"])
        self.rag_service = None

    def prepare_interview_materials(self, query: str, prepare_guide: bool = True):
        """
        Основной пайплайн: сбор → индексация → предоставление материалов
        Возвращает строку с результатами
        """
        print(f"\n{'=' * 80}")
        print(f"PREPARING INTERVIEW MATERIALS FOR: {query}")
        print('=' * 80)

        # Шаг 1: Сбор релевантных материалов
        print("\n[1/3] COLLECTING RELEVANT MATERIALS...")
        collected_docs = self.collector.collect_for_query(query)

        if not collected_docs:
            result = "No materials found for this query."
            print(result)
            return result

        print(f"Collected {len(collected_docs)} documents.")

        # Шаг 2: Индексация материалов
        print("\n[2/3] INDEXING COLLECTED MATERIALS...")
        index = self.indexer.build_index_from_documents(collected_docs)

        if not index:
            result = "Indexing failed."
            print(result)
            return result

        # Шаг 3: Инициализация RAG сервиса
        print("\n[3/3] SETTING UP RAG SERVICE...")
        self.rag_service = InterviewRAGService(
            index=index,
            config=self.config["rag"]
        )

        # Получение релевантного контекста
        print("\n" + "=" * 80)
        print("RELEVANT STUDY MATERIALS FOUND:")
        print("=" * 80)

        context = self.rag_service.get_study_context(query)

        # Проверяем, не вернулась ли ошибка
        if isinstance(context, dict) and "error" in context:
            result = f"Error: {context['error']}"
            print(result)
            return result

        if prepare_guide:
            guide = self.rag_service.generate_study_guide(query, context)
            return guide
        else:
            # Если не нужен guide, возвращаем форматированный контекст
            result = self._format_context_for_output(context)
            return result

    def _format_context_for_output(self, context):
        """Форматирует контекст для вывода"""
        output = []
        output.append("=" * 80)
        output.append("COLLECTED STUDY MATERIALS")
        output.append("=" * 80)

        if isinstance(context, dict):
            output.append(f"\nQuery: {context.get('query', 'Unknown')}")
            output.append(f"Total chunks found: {context.get('total_chunks', 0)}")
            output.append(f"Topics: {', '.join(context.get('topics', []))}")
            output.append(f"Sources: {', '.join(context.get('sources', []))}")

            output.append("\n" + "-" * 40)
            output.append("RELEVANT CONTENT:")
            output.append("-" * 40)

            for i, chunk in enumerate(context.get('chunks', []), 1):
                output.append(f"\n[{i}] Source: {chunk.get('metadata', {}).get('source', 'Unknown')}")
                output.append(f"Title: {chunk.get('metadata', {}).get('title', 'Unknown')}")
                output.append(f"Relevance: {chunk.get('score', 0):.3f}")
                output.append(f"Content:\n{chunk.get('content', 'No content')}")
                output.append("-" * 40)
        else:
            output.append(str(context))

        return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="Interview Preparation RAG System")

    parser.add_argument("--query", type=str, required=True,
                        help="Interview topic or question to prepare for")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--guide", action="store_true",
                        help="Generate a structured study guide")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode - minimal collection")
    parser.add_argument("--json", action="store_true",
                        help="Save results as JSON")

    args = parser.parse_args()

    if not args.query:
        parser.print_help()
        sys.exit(1)

    print("Initializing Interview Preparation System...")
    pipeline = InterviewPreparationPipeline(args.config)

    if args.quick:
        pipeline.config["gather"]["max_results"] = 2

    result = pipeline.prepare_interview_materials(
        query=args.query,
        prepare_guide=args.guide
    )

    if result:
        # Сохранение результатов
        query_safe = "".join(c for c in args.query if c.isalnum() or c in " _-").strip()

        if args.json:
            # Сохраняем как JSON если нужно
            output_file = f"preparation_{query_safe}.json"
            if isinstance(result, dict):
                result_data = result
            else:
                result_data = {"text": result, "query": args.query}

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
        else:
            # Сохраняем как текст
            output_file = f"preparation_{query_safe}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                if isinstance(result, dict):
                    # Если результат всё ещё словарь, преобразуем в строку
                    result_str = json.dumps(result, indent=2, ensure_ascii=False)
                    f.write(result_str)
                else:
                    f.write(result)

        print(f"\nResults saved to: {output_file}")

        # Выводим результат на экран
        print("\n" + "=" * 80)
        print("FINAL RESULT:")
        print("=" * 80)
        print(result if isinstance(result, str) else json.dumps(result, indent=2))


if __name__ == "__main__":
    main()