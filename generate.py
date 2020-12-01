from model import MusicTransformerDecoder
import datetime
from processor import decode_midi, encode_midi

max_seq = 1024
load_path = '.\\saved_model'
length = 128
save_path= 'generated.mid'

mt = MusicTransformerDecoder(loader_path=load_path)

inputs = encode_midi('dataset/midi/test.mid')

result = mt.generate(inputs[:20], length=length)

decode_midi(result, file_path=save_path)
