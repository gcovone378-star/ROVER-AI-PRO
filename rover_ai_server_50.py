"""R.O.V.E.R. AI PRO 4.1 - server locale per webcam browser + YOLO.
Esegui: python rover_ai_server.py
Poi apri http://127.0.0.1:8765 in Microsoft Edge.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import urllib.request
import urllib.error
import threading
import time

ROOT = Path(__file__).resolve().parent
HTML = ROOT / 'ROVER_AI_PRO_5_0.html'
LOCK = threading.Lock()
MODEL = None
MODEL_ERROR = None
VOICE_LOCK = threading.Lock()
VOICE_EVENTS = []
VOICE_STATUS = {'message':'in attesa'}
def voice_event(kind, text=None):
    with VOICE_LOCK:
        event = {'id':VOICE_EVENTS[-1]['id']+1 if VOICE_EVENTS else 1,'type':kind}
        if text is not None:
            event['text'] = text
        VOICE_EVENTS.append(event)
        del VOICE_EVENTS[:-20]
def voice_status(message):
    with VOICE_LOCK:
        VOICE_STATUS['message'] = message



def get_model():
    global MODEL, MODEL_ERROR
    if MODEL is None:
        if MODEL_ERROR:
            raise RuntimeError(MODEL_ERROR)
        try:
            from ultralytics import YOLO
            MODEL = YOLO('yolov8n.pt')
        except Exception as exc:
            MODEL_ERROR = str(exc)
            raise RuntimeError('Impossibile caricare YOLO: ' + MODEL_ERROR) from exc
    return MODEL


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, status, content, content_type):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if self.path in ('/', '/ROVER_AI_PRO_5_0.html'):
            if not HTML.exists():
                self.send_bytes(404, b'File HTML mancante nella cartella', 'text/plain; charset=utf-8')
                return
            self.send_bytes(200, HTML.read_bytes(), 'text/html; charset=utf-8')
        elif self.path.startswith('/voice/events'):
            from urllib.parse import urlparse, parse_qs
            after = int(parse_qs(urlparse(self.path).query).get('after',['0'])[0])
            with VOICE_LOCK:
                events = [e for e in VOICE_EVENTS if e['id']>after]
                status = VOICE_STATUS['message']
            self.send_bytes(200,json.dumps({'events':events,'status':status}).encode(),'application/json; charset=utf-8')
        elif self.path == '/health':
            try:
                with urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=2) as response:
                    available = response.status == 200
                    tags = json.loads(response.read()).get('models',[])
                    model_ready = any(x.get('name','').startswith('gemma3:4b') for x in tags)
            except Exception:
                available = False
                model_ready = False
            self.send_bytes(200,json.dumps({'ollama':available,'gemma4b':model_ready}).encode(),'application/json; charset=utf-8')
        else:
            self.send_bytes(404, b'Non trovato', 'text/plain; charset=utf-8')

    def do_POST(self):
        if self.path == '/tts':
            self.tts_request()
            return
        if self.path == '/chat':
            self.chat_request()
            return
        if self.path != '/detect':
            self.send_bytes(404, b'Non trovato', 'text/plain; charset=utf-8')
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 3_000_000:
                raise ValueError('Fotogramma vuoto o troppo grande')
            jpeg = self.rfile.read(size)
            import cv2
            import numpy as np
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('Fotogramma non valido')
            h, w = frame.shape[:2]
            start = time.monotonic()
            with LOCK:
                model = get_model()
                result = model.predict(frame, imgsz=416, conf=0.40, verbose=False, device='cpu')[0]
            objects = []
            for b in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                cls = int(b.cls[0])
                objects.append({
                    'label': str(result.names[cls]),
                    'confidence': round(float(b.conf[0]), 3),
                    'box': [max(0, min(1, x1/w)), max(0, min(1, y1/h)),
                            max(0, min(1, x2/w)), max(0, min(1, y2/h))],
                })
            response = {'objects': objects, 'inference_ms': round((time.monotonic()-start)*1000)}
            self.send_bytes(200, json.dumps(response).encode(), 'application/json; charset=utf-8')
        except Exception as exc:
            message = json.dumps({'error': str(exc)}).encode()
            self.send_bytes(500, message, 'application/json; charset=utf-8')


    def tts_request(self):
        try:
            import asyncio, tempfile, os
            import edge_tts
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<10000: raise ValueError('Testo troppo lungo')
            message=str(json.loads(self.rfile.read(size)).get('text',''))[:1400].strip()
            if not message: raise ValueError('Testo vuoto')
            fd, name=tempfile.mkstemp(suffix='.mp3');os.close(fd)
            try:
                async def make():
                    await edge_tts.Communicate(message, voice='it-IT-IsabellaNeural', rate='-5%').save(name)
                asyncio.run(make())
                self.send_bytes(200,Path(name).read_bytes(),'audio/mpeg')
            finally:
                try: os.remove(name)
                except OSError: pass
        except Exception as exc:
            self.send_bytes(503,json.dumps({'error':str(exc)}).encode(),'application/json')

    def chat_request(self):
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0 < size < 2_000_000:
                raise ValueError('Richiesta troppo grande o vuota')
            data=json.loads(self.rfile.read(size))
            question=str(data.get('message','')).strip()[:1500]
            if not question:
                raise ValueError('Domanda vuota')
            if question.lower().strip(' .!?') in ('presentati','rover presentati','ehi rover presentati'):
                answer='Sono R.O.V.E.R. AI PRO, un progetto di robotica e intelligenza artificiale ideato e sviluppato da Giuseppe Covone, della classe 5E Automazione, per l’Esame di Stato. Utilizzo YOLO per il riconoscimento di oggetti e Gemma 3, un modello linguistico sviluppato da Google, per conversare in italiano.'
                self.send_bytes(200,json.dumps({'reply':answer,'model':'identità R.O.V.E.R.'}).encode(),'application/json; charset=utf-8')
                return
            image=data.get('image')
            if image and (not isinstance(image,str) or len(image)>1_500_000):
                raise ValueError('Immagine non valida')
            history=data.get('history',[])
            messages=[{'role':'system','content':(
              'Sei R.O.V.E.R. AI PRO, assistente di un progetto scolastico di robotica ideato e sviluppato da Giuseppe Covone, classe 5E Automazione, per l’Esame di Stato. '
              'Gemma 3 è un modello linguistico di Google utilizzato nel progetto, ma Google NON è il creatore di R.O.V.E.R. Quando ti chiedono di presentarti, specifica questa distinzione. '
              'Comprendi la domanda e rispondi direttamente, senza frasi introduttive ripetitive. '
              'Per domande semplici usa 2-4 frasi; per richieste complesse spiega con esempi chiari. '
              'Non inventare fatti, misure o capacità. Se la domanda è ambigua, chiedi un chiarimento. '
              'Non dichiarare di vedere la webcam se non hai ricevuto un’immagine in questo messaggio. '
              'Se ricevi un’immagine, descrivi solo ciò che è visibile e segnala le incertezze. '
              'Non inventare stato di batteria, sensori, ESP32, motori o movimento. '
              'Sei un software in prova sul PC, non un robot fisico già collegato. '
              'Usa italiano naturale, adatto a essere pronunciato ad alta voce.'
              )}]
            if isinstance(history,list):
                for entry in history[-8:]:
                    if isinstance(entry,dict) and entry.get('role') in ('user','assistant'):
                        messages.append({'role':entry['role'],'content':str(entry.get('content',''))[:1200]})
            user={'role':'user','content':question}
            if image:user['images']=[image]
            messages.append(user)
            model='gemma3:4b'
            payload=json.dumps({'model':model,'messages':messages,'stream':False,
                'options':{'temperature':0.25,'num_predict':320}}).encode()
            req=urllib.request.Request('http://127.0.0.1:11434/api/chat',data=payload,
                headers={'Content-Type':'application/json'},method='POST')
            try:
                with urllib.request.urlopen(req,timeout=100) as response:
                    result=json.loads(response.read())
            except urllib.error.HTTPError as e:
                details=e.read().decode('utf-8','replace')[:300]
                raise RuntimeError('Ollama: '+details) from e
            answer=result.get('message',{}).get('content','').strip()
            self.send_bytes(200,json.dumps({'reply':answer,'model':model}).encode(),'application/json; charset=utf-8')
        except Exception as e:
            self.send_bytes(500,json.dumps({'error':str(e)}).encode(),'application/json; charset=utf-8')


if __name__ == '__main__':
    print('R.O.V.E.R. AI PRO 4.3 - YOLO + Gemma 3')
    print('Apri in Edge: http://127.0.0.1:8765')
    print('Per chiudere: Ctrl+C')
    print('Nota: al primo avvio YOLO potrebbe scaricare il modello yolov8n.pt.')
    ThreadingHTTPServer(('127.0.0.1', 8765), Handler).serve_forever()
