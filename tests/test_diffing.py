#!/usr/bin/env python3
from plain_language_ch_verwaltung.diffing import compute_diff_tags


def test_jargon_term_removal_is_detected():
	original = 'Gegen die Verfügung kann Einsprache erhoben werden, gemäss Verordnung.'
	simplified = 'Sie können dagegen etwas unternehmen.'

	tags = compute_diff_tags(original, simplified)

	assert 'einsprache' in tags.jargon_terms_removed
	assert 'gemäss' in tags.jargon_terms_removed


def test_jargon_term_kept_is_reported_as_remaining():
	original = 'Gegen die Verfügung kann Einsprache erhoben werden.'
	simplified = 'Sie können Einsprache gegen die Verfügung erheben.'

	tags = compute_diff_tags(original, simplified)

	assert 'einsprache' in tags.jargon_terms_remaining
	assert 'einsprache' not in tags.jargon_terms_removed


def test_sentence_splitting_is_counted():
	original = 'Dies ist ein langer Satz, der mehrere Informationen enthält und daher komplex ist.'
	simplified = 'Dies ist ein Satz. Er ist einfach.'

	tags = compute_diff_tags(original, simplified)

	assert tags.sentences_before == 1
	assert tags.sentences_after == 2
