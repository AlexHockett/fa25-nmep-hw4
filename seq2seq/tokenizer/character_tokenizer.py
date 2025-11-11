from .tokenizer import Tokenizer

import torch


class CharacterTokenizer(Tokenizer):
    def __init__(self, verbose: bool = False):
        """
        Initializes the CharacterTokenizer class for French to English translation.
        If verbose is True, prints out the vocabulary.

        We ignore capitalization.

        Implement the remaining parts of __init__ by building the vocab.
        Implement the two functions you defined in Tokenizer here. Once you are
        done, you should pass all the tests in test_character_tokenizer.py.
        """
        super().__init__()


        # Normally, we iterate through the dataset and find all unique characters. To simplify things,
        # we will use a fixed set of characters that we know will be present in the dataset.
        self.characters = """aàâæbcçdeéèêëfghiîïjklmnoôœpqrstuùûüvwxyÿz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}’•–í€óá«»… º◦©ö°äµ—ø­·òãñ―½¼γ®⇒²▪−√¥£¤ß´úª¾є™，ﬁõ  �►□′″¨³‑¯≈ˆ§‰●ﬂ⇑➘①②„≤±†✜✔➪✖◗¢ไทยếệεληνικαåşıруский 한국어汉语ž¹¿šćþ‚‛─÷〈¸⎯×←→∑δ■ʹ‐≥τ;∆℡ƒð¬¡¦βϕ▼⁄ρσ⋅≡∂≠π⎛⎜⎞ω∗"""
        self.vocab = {c:i for i, c in enumerate(self.characters)}
        self.reverse_vocab = {i:c for c, i in self.vocab.items()}

        if verbose:
            print("Vocabulary:", self.vocab)

    def encode(self, text: str) -> torch.Tensor:
        text = text.lower()
        indices = [self.vocab[c] for c in text if c in self.vocab]
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        chars = [self.reverse_vocab[int(i)] for i in tokens.tolist() if int(i) in self.reverse_vocab]
        return "".join(chars)
