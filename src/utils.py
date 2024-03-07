from fastapi.responses import HTMLResponse 
import json

class Context:
    def __init__(self, message: str = None, reason: str = None, status_code: int = None) -> None:
        self.message: str = message
        self.reason: str = reason
        self.status_code: int = status_code
        self.payload = None

    def set_payload(self, payload):
        self.payload = payload

    def set_error(self, reason: str, status_code: int = 400):
        self.reason = reason
        self.status_code = status_code

    def set_success(self, message: str, reason: str = None, status_code: int = None):
        self.message = message
        self.reason  = reason
        if status_code:
            self.status_code = status_code

    def create_success_messaga(self):
        if self.payload is not None:
            return self.payload
        
        message = self.message
        if not message:
            message = 'OK'
        return {
            "message": message
        }    

    def create_error_messaga(self):
        reason = "unnown error"
        if self.reason:
            reason = self.reason

        message = self.message
        if not message:
            message = self.reason
        
        status = 400
        if self.status_code:
            status = self.status_code

        payload = json.dumps({
            "message": message
        })    

        return HTMLResponse(content=payload, status_code=status)
