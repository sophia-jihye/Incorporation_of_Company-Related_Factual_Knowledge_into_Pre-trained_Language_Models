from pathlib import Path
import torch, linecache

class LazyLineByLineTextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, file_path, has_empty_lines=True):
        self.fin = file_path
        self.has_empty_lines = has_empty_lines
        self.tokenizer = tokenizer
        self.num_entries = self._get_n_lines(self.fin)

    def _get_n_lines(self, fin):
        with Path(fin).resolve().open(encoding='utf-8') as fhin:
            
            empty_lines = 0
            for line_idx, line in enumerate(fhin, 1):
                if line == '\n':
                    empty_lines+=1
                else:
                    pass

        return (line_idx - empty_lines) if self.has_empty_lines else line_idx

    def __getitem__(self, idx):

        if self.has_empty_lines:
            idx = idx*2

        idx += 1
        line = linecache.getline(self.fin, idx)
        line = line.rstrip()
        if line == '[EOD]': raise StopIteration
        
        # Truncates sequences at 512, does not feed the rest to the model.
        line = self.tokenizer.encode_plus(line, truncation=True, padding='max_length', max_length=512)

        return line

    def __len__(self):
        return self.num_entries
