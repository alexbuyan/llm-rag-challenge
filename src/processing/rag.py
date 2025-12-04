import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InterviewRAGService:
    def __init__(self, index, config: Dict):
        self.index = index
        self.config = config
        self.query_engine = None
        self._initialize_query_engine()

    def _initialize_query_engine(self):
        """Инициализация движка запросов"""
        try:
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=self.config.get("similarity_top_k", 4),
                response_mode="tree_summarize",
                temperature=0.1,
                verbose=True
            )
            logger.info("Query engine initialized")
        except Exception as e:
            logger.error(f"Error initializing query engine: {e}")

    def get_study_context(self, query: str) -> Dict[str, Any]:
        """Получение релевантного контекста для изучения"""
        if not self.query_engine:
            return {"error": "Query engine not initialized"}

        try:
            # Получение релевантных чанков
            retriever = self.index.as_retriever(
                similarity_top_k=self.config.get("similarity_top_k", 4)
            )

            nodes = retriever.retrieve(query)

            # Структурирование контекста
            context = {
                "query": query,
                "total_chunks": len(nodes),
                "chunks": [],
                "topics": set(),
                "sources": set()
            }

            for i, node in enumerate(nodes, 1):
                score = node.score or 0.0
                content = node.node.get_content().strip()
                metadata = node.node.metadata

                chunk_info = {
                    "id": i,
                    "score": float(score),
                    "content": content[:500] + "..." if len(content) > 500 else content,
                    "full_content": content,
                    "metadata": {
                        "title": metadata.get('title', 'Unknown'),
                        "source": metadata.get('source', 'Unknown'),
                        "url": metadata.get('url', ''),
                        "topic": metadata.get('topic', '')
                    }
                }

                context["chunks"].append(chunk_info)
                if metadata.get('topic'):
                    context["topics"].add(metadata['topic'])
                if metadata.get('source'):
                    context["sources"].add(metadata['source'])

            context["topics"] = list(context["topics"])
            context["sources"] = list(context["sources"])

            return context

        except Exception as e:
            logger.error(f"Error getting study context: {e}")
            return {"error": str(e)}

    def generate_study_guide(self, query: str, context: Dict[str, Any]) -> str:
        """Генерация структурированного руководства для изучения"""
        if "error" in context:
            return f"Error: {context['error']}"

        guide_parts = []

        # Заголовок
        guide_parts.append("=" * 80)
        guide_parts.append(f"INTERVIEW PREPARATION GUIDE: {query.upper()}")
        guide_parts.append("=" * 80)

        # Обзор
        guide_parts.append("\n📋 OVERVIEW")
        guide_parts.append("-" * 40)
        guide_parts.append(f"• Found {context['total_chunks']} relevant information chunks")
        guide_parts.append(f"• Sources: {', '.join(context['sources'])}")
        guide_parts.append(f"• Topics covered: {', '.join(context['topics'])}")

        # Ключевые вопросы для изучения
        guide_parts.append("\n❓ KEY QUESTIONS TO PREPARE")
        guide_parts.append("-" * 40)

        # Извлечение потенциальных вопросов из контекста
        key_questions = self._extract_key_questions(context)
        for i, question in enumerate(key_questions, 1):
            guide_parts.append(f"{i}. {question}")

        # Концепции для понимания
        guide_parts.append("\n💡 KEY CONCEPTS TO UNDERSTAND")
        guide_parts.append("-" * 40)

        concepts = self._extract_key_concepts(context)
        for concept in concepts:
            guide_parts.append(f"• {concept}")

        # Релевантные материалы
        guide_parts.append("\n📚 RELEVANT STUDY MATERIALS")
        guide_parts.append("-" * 40)

        for chunk in context["chunks"]:
            guide_parts.append(f"\n[Source: {chunk['metadata']['source']}]")
            guide_parts.append(f"Title: {chunk['metadata']['title']}")
            if chunk['metadata']['url']:
                guide_parts.append(f"URL: {chunk['metadata']['url']}")
            guide_parts.append(f"Relevance score: {chunk['score']:.3f}")
            guide_parts.append(f"Content:\n{chunk['content']}\n")

        # Практические советы
        guide_parts.append("\n🎯 PRACTICAL TIPS")
        guide_parts.append("-" * 40)
        guide_parts.append("1. Practice explaining each concept out loud")
        guide_parts.append("2. Create flashcards for key terms")
        guide_parts.append("3. Solve related coding problems")
        guide_parts.append("4. Prepare real-world examples")
        guide_parts.append("5. Review system design trade-offs")

        # Следующие шаги
        guide_parts.append("\n🚀 NEXT STEPS")
        guide_parts.append("-" * 40)
        guide_parts.append("1. Review the materials above")
        guide_parts.append("2. Practice with mock interviews")
        guide_parts.append("3. Update your knowledge gaps")
        guide_parts.append("4. Prepare your success stories")

        return "\n".join(guide_parts)

    def _extract_key_questions(self, context: Dict) -> List[str]:
        """Извлечение ключевых вопросов из контекста"""
        questions = []

        # Общие вопросы для собеседований
        base_questions = [
            f"What is {context['query']} and how does it work?",
            f"Explain the main components of {context['query']}",
            f"What are the advantages and disadvantages of {context['query']}?",
            f"How would you implement {context['query']} in a real system?",
            f"What are common use cases for {context['query']}?"
        ]

        # Добавление вопросов из контекста
        for chunk in context["chunks"]:
            content = chunk["full_content"].lower()

            # Поиск вопросов в тексте
            if '?' in content:
                sentences = content.split('.')
                for sentence in sentences:
                    if '?' in sentence and len(sentence.split()) > 5:
                        question = sentence.strip()
                        if question not in questions and len(questions) < 10:
                            questions.append(question[:200])  # Ограничение длины

        # Если не нашли вопросов в контексте, используем базовые
        if not questions:
            questions = base_questions[:5]

        return questions[:7]  # Ограничиваем количество

    def _extract_key_concepts(self, context: Dict) -> List[str]:
        """Извлечение ключевых концепций из контекста"""
        concepts = set()

        # Общие концепции для собеседований
        base_concepts = [
            "Time and Space Complexity",
            "System Architecture",
            "Design Patterns",
            "Best Practices",
            "Trade-offs and Optimization",
            "Scalability Considerations",
            "Security Implications"
        ]

        # Добавление из контекста
        for chunk in context["chunks"]:
            metadata = chunk["metadata"]
            if metadata.get('topic'):
                concepts.add(metadata['topic'].title())

        # Добавление базовых концепций
        for concept in base_concepts:
            concepts.add(concept)

        return list(concepts)[:10]  # Ограничиваем количество


# Класс для обратной совместимости
class RAGService:
    def __init__(self, persist_dir: str = "data/processed", use_openai_embeddings: bool = True):
        self.persist_dir = persist_dir
        self.use_openai_embeddings = use_openai_embeddings
        self.rag_service = None

    def initialize(self):
        from src.processing.index import load_existing_index
        index = load_existing_index(self.persist_dir, self.use_openai_embeddings)
        if not index:
            return False

        config = {"similarity_top_k": 4}
        self.rag_service = InterviewRAGService(index, config)
        return True

    def query(self, question: str) -> str:
        if not self.rag_service:
            if not self.initialize():
                return "Error: Failed to initialize RAG service"

        context = self.rag_service.get_study_context(question)
        return self.rag_service.generate_study_guide(question, context)