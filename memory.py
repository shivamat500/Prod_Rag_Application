import json
from pathlib import path


from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage,messages_from_dict,messages_to_dict


class MemoryManager:
    def __init__(self, memory_dir, session_id, window= 10):
        self.memory_dir = path(memory_dir)
        self.session_id = session_id
        self.window = window
        self._dir= path(memory_dir)
        self._buffer = InMemoryChatMessageHistory()
        self.load() # resume the memory from disk if available

        # Load existing memory if available
        self.load_memory()

# where to wire new memory to history
@property
def _path(self):
    return self._dir / f"{self.session_id}.json"

def add_user_message(self, message: str):
    self._buffer.add_user_message(message)
    self.save()

def add_ai_message(self, message: str):
    self._buffer.add_ai_message(message)
    self.save()

# reading from memory
@property

def all_message(self):
    return list(self._buffer.messages)

def window_messages(self):
    return self.all_message[-self.window:] if self.window else self.all_messages


def save(self):
    self._dir.mkdir(parents=True, exist_ok=True)
    payload = messages_to_dict(self._buffer.messages)
    self,_path.write_text(json.dumps(payload, indent=2, encoding ='utf-8'))

def load(self):
    if not self._path.exists():
        return
    try:
        payload = json.loads(self._path.read_text(encoding='utf-8'))
        for message in messages_from_dict(payload):
            self._buffer.add_message(message)
    except Exception as e:
        print(f"Error loading memory from {self._path}: {e}")

def clear(self):
    self._buffer.clear()
    if self._path.exists():
        self._path.unlink()  # Delete the file

