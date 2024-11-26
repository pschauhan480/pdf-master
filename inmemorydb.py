class InMemoryVectorDB:
    def __init__(self):
        self.collections = {}

    def get_or_create_collection(self, name):
        if name not in self.collections:
            self.collections[name] = Collection()
        return self.collections[name]

class Collection:
    def __init__(self):
        self.embeddings = []
        self.documents = []
        self.ids = []
        self.maxlength = 10

    def add(self, embeddings, documents, ids):
        self.embeddings.extend(embeddings)
        self.documents.extend(documents)
        self.ids.extend(ids)

    def cosine_similarity(self, vec1, vec2):
        dot_product = sum(a*b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a*a for a in vec1) ** 0.5
        magnitude2 = sum(b*b for b in vec2) ** 0.5
        if magnitude1 * magnitude2 == 0:
            return 0
        return dot_product / (magnitude1 * magnitude2)

    def query(self, query_embeddings):
        similarities = [self.cosine_similarity(query_embedding, embedding) for query_embedding in query_embeddings for embedding in self.embeddings]
        sorted_similarities = [i[0] for i in sorted(enumerate(similarities), key=lambda x:x[1])]
        # print(similarities, sorted_similarities)
        # similarities.sort()
        # best_match_index = similarities.index(max(similarities))
        # print(similarities)
        selected_documents = []
        selected_ids = []
        count = 0
        for i in sorted_similarities:
            selected_documents.append(self.documents[i])
            selected_ids.append(self.ids[i])
            if count > self.maxlength:
                break
            count += 1
        results = {
            'ids': selected_ids,
            'documents': selected_documents,
        }
        return results