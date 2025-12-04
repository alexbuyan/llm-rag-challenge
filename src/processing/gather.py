import os
import json
import requests
import arxiv
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from pathlib import Path
import time
from urllib.parse import urlparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InterviewDataCollector:
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = Path(config.get("data_dir", "data/raw"))
        self.max_results = config.get("max_results", 5)
        self.include_sources = config.get("include_sources", ["arxiv", "github"])

        self.data_dir.mkdir(parents=True, exist_ok=True)

    def collect_for_query(self, query: str) -> List[Dict]:
        """Собирает материалы по запросу пользователя"""
        query_topics = self._expand_query_to_topics(query)
        all_documents = []

        print(f"Searching for materials related to: {query}")

        # Сбор из разных источников
        if "arxiv" in self.include_sources:
            arxiv_docs = self._collect_from_arxiv(query_topics)
            all_documents.extend(arxiv_docs)
            print(f"  ArXiv: {len(arxiv_docs)} documents")

        if "github" in self.include_sources:
            github_docs = self._collect_from_github(query)
            all_documents.extend(github_docs)
            print(f"  GitHub: {len(github_docs)} documents")

        if "web" in self.include_sources:
            web_docs = self._collect_from_web(query)
            all_documents.extend(web_docs)
            print(f"  Web: {len(web_docs)} documents")

        # Сохранение собранных материалов
        if all_documents:
            query_safe = "".join(c for c in query if c.isalnum() or c in " _-").strip()
            session_dir = self.data_dir / f"session_{int(time.time())}_{query_safe[:50]}"
            session_dir.mkdir(exist_ok=True)

            session_file = session_dir / "collected_documents.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(all_documents, f, indent=2, ensure_ascii=False)

            print(f"\nTotal collected: {len(all_documents)} documents")
            print(f"Session saved to: {session_dir}")

        return all_documents

    def _expand_query_to_topics(self, query: str) -> List[str]:
        """Расширяет запрос пользователя в список тем для поиска"""
        base_topics = [query]

        # Добавление связанных тем
        topic_map = {
            "system design": ["system architecture", "scalability", "distributed systems"],
            "algorithms": ["data structures", "time complexity", "leetcode"],
            "machine learning": ["deep learning", "neural networks", "model evaluation"],
            "devops": ["docker", "kubernetes", "ci/cd", "monitoring"],
            "backend": ["api design", "databases", "caching", "authentication"],
            "frontend": ["react", "javascript", "css", "web performance"],
            "databases": ["sql", "nosql", "indexing", "transactions"]
        }

        query_lower = query.lower()
        for key, related_topics in topic_map.items():
            if key in query_lower:
                base_topics.extend(related_topics)
                break

        return base_topics[:5]  # Ограничиваем количество тем

    def _collect_from_arxiv(self, topics: List[str]) -> List[Dict]:
        """Сбор материалов с ArXiv"""
        documents = []

        for topic in topics:
            try:
                search_query = f"{topic} interview questions preparation"
                search = arxiv.Search(
                    query=search_query,
                    max_results=self.max_results,
                    sort_by=arxiv.SortCriterion.Relevance
                )

                client = arxiv.Client()
                results = list(client.results(search))

                for paper in results:
                    # Фильтрация по релевантности
                    if self._is_interview_related(paper.title, paper.summary):
                        doc = {
                            "text": paper.summary,
                            "content": paper.summary,
                            "metadata": {
                                "title": paper.title,
                                "authors": [str(a) for a in paper.authors],
                                "url": paper.entry_id,
                                "source": "arxiv",
                                "topic": topic,
                                "published": str(paper.published),
                                "categories": paper.categories
                            }
                        }
                        documents.append(doc)

            except Exception as e:
                logger.error(f"Error collecting from ArXiv for topic {topic}: {e}")

        return documents

    def _collect_from_github(self, query: str) -> List[Dict]:
        """Сбор материалов с GitHub репозиториев"""
        documents = []

        # Популярные репозитории с вопросами для собеседований
        github_repos = [
            "DopplerHQ/awesome-interview-questions",
            "yangshun/tech-interview-handbook",
            "h5bp/Front-end-Developer-Interview-Questions",
            "donnemartin/system-design-primer",
            "jwasham/coding-interview-university"
        ]

        for repo in github_repos:
            try:
                # Получение README
                readme_url = f"https://raw.githubusercontent.com/{repo}/main/README.md"
                response = requests.get(readme_url, timeout=10)

                if response.status_code == 200:
                    content = response.text

                    # Проверка на релевантность запросу
                    if self._is_relevant_to_query(content, query):
                        doc = {
                            "text": content[:5000],  # Ограничиваем размер
                            "content": content[:5000],
                            "metadata": {
                                "title": f"GitHub: {repo}",
                                "url": f"https://github.com/{repo}",
                                "source": "github",
                                "repo": repo,
                                "topic": query
                            }
                        }
                        documents.append(doc)

            except Exception as e:
                logger.error(f"Error collecting from GitHub repo {repo}: {e}")

        return documents

    def _collect_from_web(self, query: str) -> List[Dict]:
        """Сбор материалов с веб-сайтов"""
        documents = []

        # Поисковые запросы для веб-скрапинга
        search_urls = self._generate_search_urls(query)

        for url in search_urls[:3]:  # Ограничиваем количество сайтов
            try:
                response = requests.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Interview Preparation Bot)'
                })

                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')

                    # Удаление ненужных элементов
                    for tag in soup(["script", "style", "nav", "footer"]):
                        tag.decompose()

                    # Извлечение основного контента
                    text = soup.get_text(separator='\n', strip=True)

                    if len(text) > 500:  # Минимальный размер
                        if self._is_relevant_to_query(text, query):
                            doc = {
                                "text": text[:10000],  # Ограничиваем размер
                                "content": text[:10000],
                                "metadata": {
                                    "title": soup.title.string if soup.title else urlparse(url).netloc,
                                    "url": url,
                                    "source": "web",
                                    "domain": urlparse(url).netloc,
                                    "topic": query
                                }
                            }
                            documents.append(doc)

            except Exception as e:
                logger.error(f"Error collecting from web {url}: {e}")

        return documents

    def _generate_search_urls(self, query: str) -> List[str]:
        """Генерация URL для поиска"""
        base_query = query.replace(' ', '+')
        urls = [
            f"https://www.geeksforgeeks.org/tag/{query.replace(' ', '-').lower()}/",
            f"https://leetcode.com/discuss/interview-question/tag/{query.replace(' ', '-').lower()}",
            f"https://stackoverflow.com/questions/tagged/{query.replace(' ', '-').lower()}",
            f"https://www.educative.io/blog/tag/{query.replace(' ', '-').lower()}",
            f"https://www.interviewbit.com/{query.replace(' ', '-').lower()}-interview-questions/"
        ]
        return urls

    def _is_interview_related(self, title: str, abstract: str) -> bool:
        """Проверка, связан ли материал с собеседованиями"""
        keywords = ['interview', 'question', 'preparation', 'guide', 'cheatsheet',
                    'faq', 'q&a', 'common', 'popular', 'top']

        text = f"{title} {abstract}".lower()
        return any(keyword in text for keyword in keywords)

    def _is_relevant_to_query(self, content: str, query: str) -> bool:
        """Проверка релевантности контента запросу"""
        content_lower = content.lower()
        query_terms = query.lower().split()

        # Проверяем наличие терминов запроса
        matches = sum(1 for term in query_terms if term in content_lower)
        return matches >= 1  # Хотя бы одно совпадение


# Функция для обратной совместимости
def download_interview_resources(topics: List[str], max_results: int = 5, data_dir: str = "data/raw"):
    config = {
        "data_dir": data_dir,
        "max_results": max_results,
        "include_sources": ["arxiv", "github"]
    }
    collector = InterviewDataCollector(config)
    return collector.collect_for_query(topics[0] if topics else "interview questions")