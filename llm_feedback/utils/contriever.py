import torch
import numpy as np
import pickle
import dataclasses

# Contriever codebase
import src.index
import src.contriever
import src.utils
import src.slurm
import src.data
import src.normalize_text


def embed_queries(args, queries, model, tokenizer):
    model.eval()
    embeddings, batch_question = [], []
    with torch.no_grad():

        for k, q in enumerate(queries):
            if args.lowercase:
                q = q.lower()
            if args.normalize_text:
                q = src.normalize_text.normalize(q)
            batch_question.append(q)

            if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:
                encoded_batch = tokenizer.batch_encode_plus(
                    batch_question,
                    return_tensors="pt",
                    max_length=args.question_maxlength,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                output = model(**encoded_batch)
                embeddings.append(output.cpu())

                batch_question = []

    embeddings = torch.cat(embeddings, dim=0)
    print(f"Questions embeddings shape: {embeddings.size()}")

    return embeddings.numpy()


def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    for i, file_path in enumerate(embedding_files):
        print(f"Loading file {file_path}")
        with open(file_path, "rb") as fin:
            ids, embeddings = pickle.load(fin)

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    print("Data indexing completed.")


def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


class Contriever:
    def __init__(self, model, tokenizer, index, passage_id_map):
        self.model = model
        self.tokenizer = tokenizer
        self.index = index
        self.passage_id_map = passage_id_map

    def get_passages(self, query: str, top_k: int = 5):
        questions_embedding = embed_queries(ContrieverArgs(), [query], self.model, self.tokenizer)
        top_ids_and_scores = self.index.search_knn(questions_embedding, top_k)
        passage_list = []
        passage_ids, _ = top_ids_and_scores[0]
        for passage_id in passage_ids:
            passage_list.append(self.passage_id_map[passage_id])
        return passage_list

    def get_multi_passages(self, query_list, top_k: int = 5):
        questions_embedding = embed_queries(ContrieverArgs(), query_list, self.model, self.tokenizer)
        top_ids_and_scores = self.index.search_knn(questions_embedding, top_k)
        passage_list_dict = {}
        for query, (passage_ids, _) in zip(query_list, top_ids_and_scores):
            passage_list = []
            for passage_id in passage_ids:
                passage_list.append(self.passage_id_map[passage_id])
            passage_list_dict[query] = passage_list
        return passage_list_dict

    def get_multi_passages_batched(self, query_list_batched, top_k: int = 5):
        if not query_list_batched:
            return {}
        flattened_query_list = [query for query_list in query_list_batched for query in query_list]
        flattened_questions_embedding = embed_queries(
            ContrieverArgs(),
            flattened_query_list, self.model, self.tokenizer
        )
        flattened_top_ids_and_scores = self.index.search_knn(flattened_questions_embedding, top_k)
        top_ids_and_scores_batched = []
        count = 0
        for query_list in query_list_batched:
            top_ids_and_scores_batched.append(
                flattened_top_ids_and_scores[count:count+len(query_list)]
            )
            count += len(query_list)

        passage_list_batched = []
        for query_list, top_ids_and_scores in zip(query_list_batched, top_ids_and_scores_batched):
            passage_list_dict = {}
            for query, (passage_ids, _) in zip(query_list, top_ids_and_scores):
                passage_list = []
                for passage_id in passage_ids:
                    passage_list.append(self.passage_id_map[passage_id])
                passage_list_dict[query] = passage_list
            passage_list_batched.append(passage_list_dict)
        return passage_list_batched

    @classmethod
    def setup(cls, passage_path: str, index_path: str):
        model, tokenizer, _ = src.contriever.load_retriever("facebook/contriever")
        model.eval()
        model = model.cuda()
        model = model.half()

        print("Loading passages")
        passages = src.data.load_passages(passage_path)
        passage_id_map = {x["id"]: x for x in passages}

        print("Loading index")
        index = src.index.Indexer(vector_sz=768, n_subquantizers=0, n_bits=8)
        index.deserialize_from(index_path)
        return cls(
            model=model,
            tokenizer=tokenizer,
            index=index,
            passage_id_map=passage_id_map,
        )


@dataclasses.dataclass
class ContrieverArgs:
    per_gpu_batch_size: int = 64
    lowercase: bool = False
    normalize_text: bool = False
    question_maxlength: int = 512
