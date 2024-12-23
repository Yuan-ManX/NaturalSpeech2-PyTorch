import torch
from torch import Tensor
from typing import Callable, List, Optional, Tuple

from torch.nn.utils.rnn import pad_sequence

from cleaner import TextProcessor
from phonemizers.espeak_wrapper import ESpeak


# default phoneme set
# 默认音素集

# 元音
_vowels = "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ"
# 非肺部辅音
_non_pulmonic_consonants = "ʘɓǀɗǃʄǂɠǁʛ"
# 肺部辅音
_pulmonic_consonants = "pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ"
# 超音段特征（重音、语调等）
_suprasegmentals = "'̃ˈˌːˑ. ,-"
# 其他符号
_other_symbols = "ʍwɥʜʢʡɕʑɺɧʲ"
# 变音符号
_diacrilics = "ɚ˞ɫ"

# 合并所有音素
_phonemes = _vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics


# default map
# 默认语言映射字典
# 键：原始语言代码
# 值：映射后的语言代码

LANGUAGE_MAP = {
    'en-us': 'en',
    'fr-fr': 'es',
    'hi': 'hi'
}


def exists(val):
    """
    检查一个值是否存在（即不为 None）。

    Args:
        val: 需要检查的值。

    Returns:
        bool: 如果值不为 None，则返回 True；否则返回 False。
    """
    return val is not None


def default(val, d):
    """
    如果值存在（即不为 None），则返回该值；否则，返回默认值。

    这是一个常用的函数，用于在值可能缺失时提供默认值。

    Args:
        val: 需要检查的值。
        d: 默认值。

    Returns:
        如果 val 存在，则返回 val；否则返回 d。
    """
    return val if exists(val) else d


class Tokenizer:
    """
    Tokenizer 类用于将文本转换为音素序列，并提供编码和解码方法。
    """
    def __init__(
        self,
        vocab = _phonemes,
        text_cleaner: Optional[Callable] = None,
        phonemizer: Optional[Callable] = None,
        default_lang = "en-us",
        add_blank: bool = False,
        use_eos_bos = False,
        pad_id = -1
    ):
        """
        初始化 Tokenizer 实例。

        Args:
            vocab (list, optional): 词汇表，默认为预定义的音素集。
            text_cleaner (Callable, optional): 文本清理函数，可选。
            phonemizer (Callable, optional): 音素化函数，可选。
            default_lang (str, optional): 默认语言代码，默认为 "en-us"。
            add_blank (bool, optional): 是否在音素序列中添加空白符，默认为 False。
            use_eos_bos (bool, optional): 是否在音素序列开头和结尾添加特殊符号，默认为 False。
            pad_id (int, optional): 填充符的 ID，默认为 -1。
        """
        # 设置文本清理函数，默认为 TextProcessor 的 phoneme_cleaners 方法
        self.text_cleaner = default(text_cleaner, TextProcessor().phoneme_cleaners)
        # 是否添加空白符
        self.add_blank = add_blank
        # 是否使用起始和结束符号
        self.use_eos_bos = use_eos_bos
        # 填充符的 ID
        self.pad_id = pad_id

        # 词汇表
        self.vocab = vocab
        # 词汇表大小
        self.vocab_size = len(vocab)

        # 创建字符到 ID 的映射字典
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        # 创建 ID 到字符的反向映射字典
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

        # 音素化器
        self.phonemizer = phonemizer
        if not exists(self.phonemizer):
            # 如果没有提供音素化器，则使用 ESpeak 作为默认音素化器
            self.phonemizer = ESpeak(language = default_lang)

        # 设置语言
        self.language = self.phonemizer.language
        # 存储未在词汇表中找到的字符
        self.not_found_characters = []

    @property
    def espeak_language(self):
        """
        获取 ESpeak 语言代码。

        Returns:
            str or None: 如果 LANGUAGE_MAP 中存在对应的语言代码，则返回该代码；否则返回 None。
        """
        return LANGUAGE_MAP.get(self.language, None)

    def encode(self, text: str) -> List[int]:
        """Encodes a string of text as a sequence of IDs."""
        """
        将文本编码为 ID 序列。

        Args:
            text (str): 输入的文本字符串。

        Returns:
            List[int]: 对应的 ID 序列。
        """
        # 初始化 ID 列表
        token_ids = []
        for char in text:
            try:
                # 查找字符对应的 ID
                idx = self.char_to_id[char]
                # 添加到 ID 列表中
                token_ids.append(idx)
            except KeyError:
                # discard but store not found characters
                # 如果字符未在词汇表中找到，则丢弃并记录
                if char not in self.not_found_characters:
                    self.not_found_characters.append(char)
                    print(text)
                    print(f" [!] Character {repr(char)} not found in the vocabulary. Discarding it.")
        # 返回编码后的 ID 序列
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decodes a sequence of IDs to a string of text."""
        """
        将 ID 序列解码为文本。

        Args:
            token_ids (List[int]): 输入的 ID 序列。

        Returns:
            str: 对应的文本字符串。
        """
        # 初始化文本字符串
        text = ""
        for token_id in token_ids:
            # 将每个 ID 转换为字符并添加到文本中
            text += self.id_to_char[token_id]
        # 返回解码后的文本
        return text

    def text_to_ids(
        self,
        text: str,
        language: str = None
    ) -> Tuple[List[int], str, str]:
        """Converts a string of text to a sequence of token IDs.

        Args:
            text(str):
                The text to convert to token IDs.

            language(str):
                The language code of the text. Defaults to None.

        TODO:
            - Add support for language-specific processing.

        1. Text normalizatin
        2. Phonemization (if use_phonemes is True)
        3. Add blank char between characters
        4. Add BOS and EOS characters
        5. Text to token IDs
        """
        """
        将文本字符串转换为音素 ID 序列。

        Args:
            text (str):
                要转换为音素 ID 的文本。

            language (str):
                文本的语言代码。默认为 None。

        TODO:
            - 添加对特定语言处理的支持。

        处理步骤:
            1. 文本规范化
            2. 音素化（如果 use_phonemes 为 True）
            3. 在字符之间添加空白字符
            4. 添加 BOS 和 EOS 字符
            5. 文本转换为音素 ID
        """
        # 如果未提供语言代码，则使用默认的语言代码

        language = default(language, self.espeak_language)

        cleaned_text = None
        if self.text_cleaner is not None:
            # 使用文本清理函数清理文本
            text = self.text_cleaner(text, language=language)
            # 保存清理后的文本
            cleaned_text = text
        
        # 使用音素化器将文本转换为音素序列
        phonemized = self.phonemizer.phonemize(text, separator="", language=language)
        if self.add_blank:
            # 如果需要添加空白字符，则在音素之间插入空白字符
            phonemized = self.intersperse_blank_char(phonemized, True)
        if self.use_eos_bos:
            # 如果需要添加起始和结束符号，则在音素序列的开头和结尾添加 BOS 和 EOS 字符
            phonemized = self.pad_with_bos_eos(phonemized)

        # 将处理后的音素序列编码为 ID 序列
        return self.encode(phonemized), cleaned_text, phonemized

    def texts_to_tensor_ids(self, texts: List[str], language: str = None) -> Tensor:
        """
        将多个文本字符串转换为填充后的音素 ID 张量。

        Args:
            texts (List[str]): 要转换的文本字符串列表。
            language (str, optional): 文本的语言代码。默认为 None。

        Returns:
            torch.Tensor: 填充后的音素 ID 张量。
        """
        all_ids = []

        for text in texts:
            # 对每个文本进行编码
            ids, *_ = self.text_to_ids(text, language = language)
            # 将 ID 列表转换为张量并添加到列表中
            all_ids.append(torch.tensor(ids))
        # 使用填充值将所有张量填充到相同的长度，并堆叠成一个批量张量
        return pad_sequence(all_ids, batch_first = True, padding_value = self.pad_id)

    def ids_to_text(self, id_sequence: List[int]) -> str:
        """Converts a sequence of token IDs to a string of text."""
        """
        将音素 ID 序列转换为文本字符串。

        Args:
            id_sequence (List[int]): 要转换的音素 ID 序列。

        Returns:
            str: 对应的文本字符串。
        """
        return self.decode(id_sequence)

    def pad_with_bos_eos(self, char_sequence: List[str]):
        """Pads a sequence with the special BOS and EOS characters."""
        """
        在字符序列的开头和结尾添加特殊字符 BOS 和 EOS。

        Args:
            char_sequence (List[str]): 要填充的字符序列。

        Returns:
            List[str]: 填充后的字符序列。
        """
        return [self.characters.bos] + list(char_sequence) + [self.characters.eos]

    def intersperse_blank_char(self, char_sequence: List[str], use_blank_char: bool = False):
        """Intersperses the blank character between characters in a sequence.

        Use the ```blank``` character if defined else use the ```pad``` character.
        """
        """
        在字符序列的每个字符之间插入空白字符。

        如果定义了 ```blank``` 字符，则使用它；否则，使用 ```pad``` 字符。

        Args:
            char_sequence (List[str]): 要插入空白字符的字符序列。
            use_blank_char (bool, optional): 是否使用空白字符。默认为 False。

        Returns:
            List[str]: 插入空白字符后的字符序列。
        """
        # 根据参数选择使用空白字符还是填充字符
        char_to_use = self.characters.blank if use_blank_char else self.characters.pad
        # 初始化结果列表，长度为 (2 * len(char_sequence) + 1)
        result = [char_to_use] * (len(char_sequence) * 2 + 1)
        # 在奇数索引位置插入原始字符
        result[1::2] = char_sequence
        return result


if __name__ == "__main__":

    # 初始化文本处理器
    txt_cleaner = TextProcessor()

    # 创建一个 Tokenizer 实例，使用默认的音素词汇表、文本清理函数和英语的 ESpeak 音素化器
    tokenizer = Tokenizer(vocab = _phonemes, text_cleaner = txt_cleaner.phoneme_cleaners, phonemizer = ESpeak(language="en-us"))
    # 将英文文本转换为音素 ID 序列并打印结果
    print(tokenizer.text_to_ids("Hello, Mr. Example, this is 9:30 am and  my number is 30.", language="en"))
    
    # 创建一个新的 Tokenizer 实例，使用默认的音素词汇表、文本清理函数和法语的 ESpeak 音素化器
    tokenizer = Tokenizer(vocab = _phonemes, text_cleaner = txt_cleaner.phoneme_cleaners, phonemizer = ESpeak(language="fr-fr"))
    # 将西班牙语文本转换为音素 ID 序列并打印结果
    print(tokenizer.text_to_ids("Hola, Sr. Ejemplo, son las 9:30 am y mi número es el 30.", language="es"))
    
    # 创建一个新的 Tokenizer 实例，使用默认的音素词汇表、文本清理函数和印地语的 ESpeak 音素化器
    tokenizer = Tokenizer(vocab = _phonemes, text_cleaner = txt_cleaner.phoneme_cleaners, phonemizer = ESpeak(language="hi"))
    # 将印地语文本转换为音素 ID 序列并打印结果
    print(tokenizer.text_to_ids("हैलो, मिस्टर उदाहरण, यह सुबह 9:30 बजे है और मेरा नंबर 30 है।", language="hi"))
