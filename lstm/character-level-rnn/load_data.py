def load_data(filename, seq_length):
    raw_text = open(filename, 'r', encoding='utf-8').read().lower()
    chars = sorted(list(set(raw_text)))
    char_to_int = {c: i for i, c in enumerate(chars)}
    dataX, dataY = [], []
    for i in range(0, len(raw_text) - seq_length):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    return dataX, dataY, char_to_int, len(chars), raw_text