#!/usr/bin/env python3
from evaluate import compute_readability


def test_simple_text_scores_easier_than_complex_text():
	simple = 'Der Hund ist müde. Er schläft im Bett. Das ist schön.'
	complex_text = (
		'Die Verwaltungsrechtspflegebeschwerde gegen die Vollzugsverordnung ist innert der '
		'gesetzlich vorgeschriebenen Frist bei der zuständigen Rekursinstanz einzureichen.'
	)

	simple_scores = compute_readability(simple)
	complex_scores = compute_readability(complex_text)

	assert simple_scores.wstf < complex_scores.wstf
	assert simple_scores.lix < complex_scores.lix
	assert simple_scores.zix > complex_scores.zix  # zix is inverted: higher = easier


def test_empty_text_does_not_crash():
	scores = compute_readability('')
	assert scores.n_words == 0
