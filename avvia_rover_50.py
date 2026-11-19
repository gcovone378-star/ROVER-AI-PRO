"""R.O.V.E.R. 5.0: desktop WebView2 + microfono Vosk offline."""
from threading import Thread
from pathlib import Path
import socket, time, json, queue
from http.server import ThreadingHTTPServer
from rover_ai_server_50 import Handler, voice_event, voice_status

ROOT=Path(__file__).resolve().parent

def microphone():
    try:
        from vosk import Model, KaldiRecognizer, SetLogLevel
        import sounddevice as sd
        SetLogLevel(-1)
        paths=[ROOT/'vosk-model-small-it-0.22',ROOT/'vosk-model-it-0.22']
        paths += [p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith('vosk-model') and 'it' in p.name]
        model_dir=next((p for p in paths if (p/'am').is_dir() and (p/'conf').is_dir()),None)
        if not model_dir:
            voice_status('modello italiano Vosk non trovato nella cartella ROVER_AI')
            return
        voice_status('caricamento Vosk...')
        model=Model(str(model_dir))
        # Prova prima il dispositivo predefinito, poi gli altri ingressi compatibili.
        devices=sd.query_devices()
        candidates=[]
        try:
            default_idx=int(sd.default.device[0])
            if default_idx>=0:candidates.append(default_idx)
        except Exception:pass
        candidates += [i for i,d in enumerate(devices) if d['max_input_channels']>0 and i not in candidates]
        errors=[]
        stream=None
        for idx in candidates:
            d=devices[idx]
            rate=int(d['default_samplerate'] or 16000)
            try:
                sd.check_input_settings(device=idx,channels=1,dtype='int16',samplerate=rate)
                audio=queue.Queue(maxsize=30)
                def callback(indata,frames,t,st):
                    try:audio.put_nowait(bytes(indata))
                    except queue.Full:pass
                stream=sd.RawInputStream(device=idx,samplerate=rate,blocksize=4000,dtype='int16',channels=1,callback=callback)
                stream.start()
                voice_status('in ascolto su '+d['name']+' · dì Ehi Rover')
                print('MICROFONO ATTIVO:',idx,d['name'],rate,flush=True)
                break
            except Exception as exc:
                errors.append(str(idx)+' '+d['name']+': '+str(exc))
                stream=None
        if stream is None:
            raise RuntimeError('Nessun ingresso microfono utilizzabile. '+ ' | '.join(errors[:3]))
        recognizer=KaldiRecognizer(model,rate)
        wake_words=('ehi rover','hey rover','ei rover','e rover','ehi rovere','hei rover')
        awaiting_command=False
        command_deadline=0.0
        last_wake=0.0
        try:
            while True:
                try:
                    data=audio.get(timeout=2)
                except queue.Empty:
                    continue
                is_final=recognizer.AcceptWaveform(data)
                result=json.loads(recognizer.Result() if is_final else recognizer.PartialResult())
                phrase=' '.join(result.get('text' if is_final else 'partial','').lower().split())
                now=time.monotonic()
                if awaiting_command and now>command_deadline:
                    awaiting_command=False
                    voice_status('in ascolto · dì Ehi Rover')
                    recognizer.Reset()
                    continue
                if not phrase:
                    continue
                if not awaiting_command:
                    matched=next((word for word in wake_words if word in phrase),None)
                    if matched and now-last_wake>3:
                        last_wake=now
                        voice_event('wake')
                        awaiting_command=True
                        command_deadline=now+14
                        voice_status('Ehi Rover riconosciuto · pronuncia la domanda')
                        # Se la frase completa contiene anche il comando, elaboralo subito.
                        remainder=phrase.split(matched,1)[1].strip(' ,.!?')
                        if is_final and remainder:
                            voice_event('command',remainder)
                            awaiting_command=False
                            voice_status('domanda ricevuta · dì Ehi Rover per continuare')
                        recognizer.Reset()
                elif is_final:
                    command=phrase
                    for word in wake_words:
                        if command.startswith(word):
                            command=command[len(word):].strip(' ,.!?')
                            break
                    if command:
                        voice_event('command',command)
                        voice_status('domanda ricevuta · dì Ehi Rover per continuare')
                        awaiting_command=False
                        recognizer.Reset()
        finally:
            stream.stop();stream.close()
    except Exception as exc:
        voice_status('errore microfono: '+str(exc))
        print('MICROFONO:',exc,flush=True)

def main():
    try: import webview
    except ImportError:
        print('Installa pywebview: python -m pip install pywebview')
        return
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',8765))==0:
            print('Porta 8765 occupata. Chiudi il vecchio R.O.V.E.R.')
            return
    Thread(target=lambda:ThreadingHTTPServer(('127.0.0.1',8765),Handler).serve_forever(),daemon=True).start()
    Thread(target=microphone,daemon=True).start()
    time.sleep(.5)
    webview.create_window('R.O.V.E.R. AI PRO 5.0 | Industrial Command Center','http://127.0.0.1:8765',width=1440,height=900,min_size=(1024,720),background_color='#071522')
    webview.start(gui='edgechromium',debug=False)
if __name__=='__main__':main()
