from model import MusicTransformerDecoder
import datetime
from processor import decode_midi, encode_midi

def main():
    load_path = 'result/result_model'
    length = 2048
    save_path= 'generated.mid'

    mt = MusicTransformerDecoder(loader_path=load_path)
    inputs = encode_midi('Midiset/test.mid')
    result = mt.generate(inputs[:20], length=length)
    decode_midi(result, file_path=save_path)
    
    
if __name__ == "__main__":
    main()
