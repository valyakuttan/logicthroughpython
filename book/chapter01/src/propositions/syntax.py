# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2021
# File name: propositions/syntax.py

"""Syntactic handling of propositional formulas."""

from __future__ import annotations
from functools import lru_cache
from typing import Mapping, Optional, Set, Tuple
from itertools import takewhile

from logic_utils import frozen, memoized_parameterless_method

@lru_cache(maxsize=100) # Cache the return value of is_variable
def is_variable(string: str) -> bool:
    """Checks if the given string is an atomic proposition.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is an atomic proposition, ``False``
        otherwise.
    """
    return string[0] >= 'p' and string[0] <= 'z' and \
        (len(string) == 1 or string[1:].isdigit())

@lru_cache(maxsize=100) # Cache the return value of is_constant
def is_constant(string: str) -> bool:
    """Checks if the given string is a constant.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a constant, ``False`` otherwise.
    """
    return string == 'T' or string == 'F'

@lru_cache(maxsize=100) # Cache the return value of is_unary
def is_unary(string: str) -> bool:
    """Checks if the given string is a unary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a unary operator, ``False`` otherwise.
    """
    return string == '~'

@lru_cache(maxsize=100) # Cache the return value of is_binary
def is_binary(string: str) -> bool:
    """Checks if the given string is a binary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a binary operator, ``False`` otherwise.
    """
    return string == '&' or string == '|' or string == '->'
    # For Chapter 3:
    # return string in {'&', '|',  '->', '+', '<->', '-&', '-|'}

@lru_cache(maxsize=100) # Cache the return value of split_str
def split_str(string):
    """
    """
    prefix, rest = (string[:2], string[2:]) if \
        (string and string[0] == '-') else \
        (string[:1], string[1:])

    if prefix and is_variable(prefix):
        ds = ''.join(takewhile(lambda d: d.isdigit(),
                               rest))
        m = len(ds)
        prefix, rest = (prefix + rest[:m]), rest[m:]

    return prefix, rest

@frozen
class Formula:
    """An immutable propositional formula in tree representation, composed from
    atomic propositions, and operators applied to them.

    Attributes:
        root (`str`): the constant, atomic proposition, or operator at the root
            of the formula tree.
        first (`~typing.Optional`\\[`Formula`]): the first operand to the root,
            if the root is a unary or binary operator.
        second (`~typing.Optional`\\[`Formula`]): the second operand to the
            root, if the root is a binary operator.
    """
    root: str
    first: Optional[Formula]
    second: Optional[Formula]

    def __init__(self, root: str, first: Optional[Formula] = None,
                 second: Optional[Formula] = None):
        """Initializes a `Formula` from its root and root operands.

        Parameters:
            root: the root for the formula tree.
            first: the first operand to the root, if the root is a unary or
                binary operator.
            second: the second operand to the root, if the root is a binary
                operator.
        """
        if is_variable(root) or is_constant(root):
            assert first is None and second is None
            self.root = root
        elif is_unary(root):
            assert first is not None and second is None
            self.root, self.first = root, first
        else:
            assert is_binary(root)
            assert first is not None and second is not None
            self.root, self.first, self.second = root, first, second

    @memoized_parameterless_method
    def __repr__(self) -> str:
        """Computes the string representation of the current formula.

        Returns:
            The standard string representation of the current formula.
        """
        # Task 1.1
        if is_constant(self.root) or is_variable(self.root):
            return self.root
        elif is_unary(self.root):
            return ''.join(map(str, [self.root, self.first]))
        elif is_binary(self.root):
            return ''.join(map(str, ['(', self.first, self.root,
                                     self.second, ')']))

    def __eq__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is a `Formula` object that equals the
            current formula, ``False`` otherwise.
        """
        return isinstance(other, Formula) and str(self) == str(other)

    def __ne__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is not a `Formula` object or does not
            equal the current formula, ``False`` otherwise.
        """
        return not self == other

    def __hash__(self) -> int:
        return hash(str(self))

    @memoized_parameterless_method
    def variables(self) -> Set[str]:
        """Finds all atomic propositions (variables) in the current formula.

        Returns:
            A set of all atomic propositions used in the current formula.
        """
        # Task 1.2
        def vars_with_acc(f: Formula, acc: Set[str]) -> Set[str]:
            if is_constant(f.root):
                return acc
            elif is_variable(f.root):
                acc.add(f.root)
                return acc
            elif is_unary(f.root):
                return vars_with_acc(f.first, acc)
            else:
                bcc = vars_with_acc(f.first, acc)
                return vars_with_acc(f.second, bcc)

        return vars_with_acc(self, set())

    @memoized_parameterless_method
    def operators(self) -> Set[str]:
        """Finds all operators in the current formula.

        Returns:
            A set of all operators (including ``'T'`` and ``'F'``) used in the
            current formula.
        """
        # Task 1.3
        def operators_with_acc(f: Formula, acc: Set[str]) -> Set[str]:
            if is_variable(f.root):
                return acc
            elif is_constant(f.root):
                acc.add(f.root)
                return acc
            elif is_unary(f.root):
                acc.add(f.root)
                return operators_with_acc(f.first, acc)
            else:
                acc.add(f.root)
                bcc = operators_with_acc(f.first, acc)
                return operators_with_acc(f.second, bcc)

        return operators_with_acc(self, set())

    @staticmethod
    def _parse_prefix(string: str) -> Tuple[Optional[Formula], str]:
        """Parses a prefix of the given string into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A pair of the parsed formula and the unparsed suffix of the string.
            If the given string has as a prefix a variable name (e.g.,
            ``'x12'``) or a unary operator follows by a variable name, then the
            parsed prefix will include that entire variable name (and not just a
            part of it, such as ``'x1'``). If no prefix of the given string is a
            valid standard string representation of a formula then returned pair
            should be of ``None`` and an error message, where the error message
            is a string with some human-readable content.
        """
        # Task 1.4
        prefix, rest = split_str(string)
        if prefix and (is_constant(prefix) or is_variable(prefix)):
            return Formula(prefix), rest

        elif is_unary(prefix):
            ff, rr = Formula._parse_prefix(rest)
            if ff is None:
                return ff, rr
            else:
                return Formula(prefix, ff), rr

        elif prefix == '(':
            ff, rr = Formula._parse_prefix(rest)
            if ff is None:
                return ff, rr

            bop, rem = split_str(rr)
            if not is_binary(bop):
                return (None, 'Unexpected symbol {} in {}'.
                        format(bop, rem))

            ss, tt = Formula._parse_prefix(rem)
            if ss is None:
                return ss, tt

            if not tt.startswith(')'):
                return None, 'Unexpected symbol {} in {}'.format(tt[:1], tt)

            return Formula(bop, ff, ss), tt[1:]

        else:
            return None, "Unexpected input {}".format(string)

    @staticmethod
    def is_formula(string: str) -> bool:
        """Checks if the given string is a valid representation of a formula.

        Parameters:
            string: string to check.

        Returns:
            ``True`` if the given string is a valid standard string
            representation of a formula, ``False`` otherwise.
        """
        # Task 1.5
        f, r = Formula._parse_prefix(string)
        return ((f is not None) and (r == ''))

    @staticmethod
    def parse(string: str) -> Formula:
        """Parses the given valid string representation into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A formula whose standard string representation is the given string.
        """
        assert Formula.is_formula(string)
        # Task 1.6
        return Formula._parse_prefix(string)[0]

# Optional tasks for Chapter 1

    def polish(self) -> str:
        """Computes the polish notation representation of the current formula.

        Returns:
            The polish notation representation of the current formula.
        """
        # Optional Task 1.7
        if is_constant(self.root) or is_variable(self.root):
            return self.root
        elif is_unary(self.root):
            return ''.join([self.root, self.first.polish()])
        else:
            return ''.join([self.root, self.first.polish(),
                            self.second.polish()])

    def _parse_prefix_polish(string: str) -> Tuple[Optional[Formula], str]:
        """
        """
        prefix, rest = split_str(string)
        if prefix and (is_constant(prefix) or is_variable(prefix)):
            return Formula(prefix), rest

        elif is_unary(prefix):
            ff, rr = Formula._parse_prefix_polish(rest)
            if ff is None:
                return ff, rr
            else:
                return Formula(prefix, ff), rr

        elif is_binary(prefix):
            ff, rr = Formula._parse_prefix_polish(rest)
            if ff is None:
                return ff, rr
            ss, tt = Formula._parse_prefix_polish(rr)
            if ss is None:
                return ss, tt
            return Formula(prefix, ff, ss), tt

        else:
            return None, "Unexpected input {}".format(string)

    @staticmethod
    def parse_polish(string: str) -> Formula:
        """Parses the given polish notation representation into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A formula whose polish notation representation is the given string.
        """
        # Optional Task 1.8
        return Formula._parse_prefix_polish(string)[0]

    def substitute_variables(
            self, substitution_map: Mapping[str, Formula]) -> Formula:
        """Substitutes in the current formula, each variable `v` that is a key
        in `substitution_map` with the formula `substitution_map[v]`.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.

        Returns:
            The formula resulting from performing all substitutions. Only
            variables originating in the current formula are substituted (i.e.,
            variables originating in one of the specified substitutions are not
            subjected to additional substitutions).

        Examples:
            >>> Formula.parse('((p->p)|r)').substitute_variables(
            ...     {'p': Formula.parse('(q&r)'), 'r': Formula.parse('p')})
            (((q&r)->(q&r))|p)
        """
        for variable in substitution_map:
            assert is_variable(variable)
        # Task 3.3

    def substitute_operators(self, substitution_map: Mapping[str, Formula]) -> \
        Formula:
        """Substitutes in the current formula, each constant or operator `op`
        that is a key in `substitution_map` with the formula
        `substitution_map[op]` applied to its (zero or one or two) operands,
        where the first operand is used for every occurrence of ``'p'`` in the
        formula and the second for every occurrence of ``'q'``.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.

        Returns:
            The formula resulting from performing all substitutions. Only
            operators originating in the current formula are substituted (i.e.,
            operators originating in one of the specified substitutions are not
            subjected to additional substitutions).

        Examples:
            >>> Formula.parse('((x&y)&~z)').substitute_operators(
            ...     {'&': Formula.parse('~(~p|~q)')})
            ~(~~(~x|~y)|~~z)
        """
        for operator in substitution_map:
            assert is_binary(operator) or is_unary(operator) or \
                   is_constant(operator)
            assert substitution_map[operator].variables().issubset({'p', 'q'})
        # Task 3.4
