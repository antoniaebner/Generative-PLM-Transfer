"""
Copyright (C) Tristan Bepler - All Rights Reserved
Author: Tristan Bepler <tbepler@gmail.com>
"""
# Original code from PoET under MIT License.
# Copyright (c) 2023 OpenProteinAI

# Modified by Antonia Ebner, 2026:
#   - added translation to another token id mapping

from __future__ import division, print_function

import numpy as np
import torch
from src.models.protxlstm.utils import ID_TO_AA


class Alphabet:
    def __init__(self, chars, encoding=None, mask=False, missing=255):
        self.chars = np.frombuffer(chars, dtype=np.uint8)
        self.encoding = np.zeros(256, dtype=np.uint8) + missing
        if encoding is None:
            self.encoding[self.chars] = np.arange(len(self.chars))
            self.size = len(self.chars)
        else:
            self.encoding[self.chars] = encoding
            self.size = encoding.max() + 1
        self.mask = mask
        if mask:
            self.size -= 1

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return chr(self.chars[i])

    def encode(self, x):
        """encode a byte string into alphabet indices"""
        x = np.frombuffer(x, dtype=np.uint8)
        return self.encoding[x]

    def decode(self, x):
        """decode index array, x, to byte string of this alphabet"""
        string = self.chars[x]
        return string.tobytes()

    def unpack(self, h, k):
        """unpack integer h into array of this alphabet with length k"""
        n = self.size
        kmer = np.zeros(k, dtype=np.uint8)
        for i in reversed(range(k)):
            c = h % n
            kmer[i] = c
            h = h // n
        return kmer

    def get_kmer(self, h, k):
        """retrieve byte string of length k decoded from integer h"""
        kmer = self.unpack(h, k)
        return self.decode(kmer)


DNA = Alphabet(b"ACGT")


class Uniprot21(Alphabet):
    def __init__(
        self,
        mask=False,
        include_gap=False,
        include_startstop=False,
        distinct_startstop=False,
    ):
        chars = b"ARNDCQEGHILKMFPSTWYV"
        gap_token = start_token = stop_token = -1
        if include_gap:
            chars = chars + b"-"
            gap_token = len(chars) - 1
        if include_startstop:
            chars = chars + b"*"
            start_token = stop_token = len(chars) - 1
        if distinct_startstop:
            chars = chars + b"$"
            stop_token = len(chars) - 1
        self.distinct_startstop = distinct_startstop
        # add the synonym tokens
        mask_token = len(chars)
        chars = chars + b"XOUBZ"

        encoding = np.arange(len(chars))
        encoding[mask_token + 1 :] = [
            11,
            4,
            mask_token,
            mask_token,
        ]  # encode 'OUBZ' as synonyms
        missing = mask_token

        super(Uniprot21, self).__init__(
            chars, encoding=encoding, mask=mask, missing=missing
        )

        self.gap_token = gap_token
        self.start_token = start_token
        self.stop_token = stop_token
        self.mask_token = mask_token
        assert (
            include_gap and include_startstop and distinct_startstop
        ), "Translation only implemented for include_gap=True and include_startstop=True and distinct_startstop=True "

        symbol_conversion = {
            "<cls>": chars.decode()[start_token],
            "<pad>": chars.decode()[missing],
            "<eos>": chars.decode()[stop_token],
            "<unk>": chars.decode()[missing],
            ".": chars.decode()[gap_token],
            "-": chars.decode()[gap_token],
            "<null_1>": chars.decode()[missing],
            "<mask>": chars.decode()[mask_token],
            "<mask-1>": chars.decode()[mask_token],
            "<mask-2>": chars.decode()[mask_token],
            "<mask-3>": chars.decode()[mask_token],
            "<mask-4>": chars.decode()[mask_token],
            "<mask-5>": chars.decode()[mask_token],
        }

        self.translation = torch.empty((len(ID_TO_AA),), dtype=torch.uint8)
        for k, v in ID_TO_AA.items():
            encoded = symbol_conversion.get(v, v).encode()
            self.translation[k] = torch.from_numpy(
                self.encoding[np.frombuffer(encoded, dtype=np.uint8)]
            )

    def translate(self, x):
        # translate from AA_TO_ID to this alphabet
        segments = np.split(x, np.where(x == 0)[0])[1:]  # first segment is always empty
        if len(segments) < 1:
            print(segments)
            return ([], [])
        # remove empty segments (and add <eos> for distinct_startstop)
        if self.distinct_startstop:
            segments = [torch.cat((seg, torch.LongTensor([2]))) for seg in segments]

        if len(segments) < 1:
            print(segments)
        x_ = torch.cat(segments)
        seg_lengths = [len(seg) for seg in segments]
        return (self.translation[x_], seg_lengths)


class SDM12(Alphabet):
    """
    A D KER N TSQ YF LIVM C W H G P

    See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2732308/#B33
    "Reduced amino acid alphabets exhibit an improved sensitivity and selectivity in fold assignment"
    Peterson et al. 2009. Bioinformatics.
    """

    def __init__(self, mask=False):
        chars = alphabet = b"ADKNTYLCWHGPXERSQFIVMOUBZ"
        groups = [
            b"A",
            b"D",
            b"KERO",
            b"N",
            b"TSQ",
            b"YF",
            b"LIVM",
            b"CU",
            b"W",
            b"H",
            b"G",
            b"P",
            b"XBZ",
        ]
        groups = {c: i for i in range(len(groups)) for c in groups[i]}
        encoding = np.array([groups[c] for c in chars])
        super(SDM12, self).__init__(chars, encoding=encoding, mask=mask)


SecStr8 = Alphabet(b"HBEGITS ")
