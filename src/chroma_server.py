from chromadb import HttpClient, Collection
from interfaces import VectorDBInterface
from utils import Context

class ChromaDBServer(VectorDBInterface):

    def __init__(self, host: str = None, port: int = None, url: str = None, collection: str = "default", parameters: dict = ...) -> None:
        super().__init__(host, port, url, collection, parameters)
        self.cdb_client: HttpClient = None
        self.cdb_collection: Collection = None
        

    def is_valid(self) -> bool:
        try:
            if not self.host:
                return False
            
            client = HttpClient(host=self.host, port=self.port)
            if not client:
                return False
            
            coll_name = self.collection
            if not coll_name:
                coll_name = "default"
            
            coll = client.get_or_create_collection(coll_name)
            if not coll:
                return False
            
            self.cdb_client = client
            self.cdb_collection = coll
            self.collection = coll_name
            print(f"Connected to chromadb {self.host}:{self.port}/{self.collection}")
            return True

        except Exception as exc:
            print(f"Error while connecting to chromadb server: {self.host}:{self.port}")
            return False
        
    def count(self, context: Context) -> bool:
        return self.cdb_collection.count()
