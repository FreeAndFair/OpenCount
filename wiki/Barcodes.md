# Hart #

Hart ballots contain three barcodes, in the upper-left, lower-left, and lower-right.  Each barcode is a 1-D barcode, with symbols encoded in [Interleaved 2 of 5](http://en.wikipedia.org/wiki/Interleaved_2_of_5) format.

The upper-left barcode encodes a 14-digit string, structured as follows:
  * Digit 0: Sheet (i.e. '1' for pages 1-2, '2' for page 3)
  * Digits 1-6: Precinct index (an index into the set of precinct numbers)
  * Digit 7: Unused (always 0)
  * Digit 8: Page
  * Digit 9: Language
  * Digits 10-11: Party
  * Digits 12-13: Checksum

I don't know what the lower-left barcode encodes.

The lower-right barcode encodes an election identifier, which should be the same for all ballots within a single election.

The checksum digits are computed as follows:
  * Denote the first 12 digits by _a_, _b_, _c_, .., _k_, _l_ (each variable represents a single digit).
  * Compute 82 _a_ + 47 _b_ + 92 _c_ + 48 _d_ + 63 _e_ + 16 _f_ + 21 _g_ + 70 _h_ + 7 _i_ + 88 _j_ + 67 _k_ + 94 _l_, and reduce modulo 97.
  * If the result is 0, convert it to 97; if the result is 1, convert it to 98.
The result of this sequence of steps is the checksum.  (This has been verified for the upper-left barcode.  We haven't checked if the same checksum is used in the other barcodes.  Notice that each weight is 68 times the previous one, modulo 97.  For example, 68\*82 % 97 = 47, and so on.)

# Diebold (Premier) #

There are barcodes at the bottom of the front side of the page and the bottom of the back side of the page.  Each one is 32 bits long, with a 1 bit indicated by a black rectangle and a 0 bit by an empty (white) area.  The most significant bit is on the left side of the ballot.

The structure of the marks on the front side:
  * Bits 0-1: Checksum: the number of 1's in bits 2-31, reduced modulo 4
  * Bits 2-14: Batch or precinct number
  * Bits 15-27: Card number (CardRotID)
  * Bits 28-30: Sequence number (always 0)
  * Bit 31: Start bit (always 1)
Higher-numbered bits are most significant (e.g., the precinct number is encoded as a 13-bit value, where bit 14 is the most significant bit and bit 2 is the least significant).

The marks on the back side:
  * Bits 0-4: Election day of month (1..31)
  * Bits 5-8: Election month (1..12)
  * Bits 9-15: Election year (2 digits)
  * Bits 16-20: Election type
  * Bits 21-31: Ender code (always binary 01111011110)

Acknowledgement: Thanks to Harri Hursti for reverse-engineering these barcodes.

# Sequoia #

There are two 8-bit codes, one on the upper left and one on the upper right.  These appear only on the front side of the ballot.

The structure of the code on the left is:
  * Bits 0-3: party (bit 0 is MSB)
  * Bits 4-7: always zero (?)
Bit 0 is at the top.

The code on the right is:
  * Bit 0-7: ballot layout identifier, in binary (bit 0 is the most significant bit)

The ballot layout identifier is also printed in decimal printed on the lower right of the ballot.  It appears that this uniquely determines the layout of information on ballot (e.g., the order and meaning of all voting targets, contests, etc.).  Voting targets will not necessarily occur in the exact same location on all ballots with the same ballot layout identifier, but they will appear in the same order within each column.

Note that the precinct number is not encoded in these codes.  It appears that it is overprinted separately, in decimal, in the upper-left, and is not encoded anywhere in any barcode or timing mark.