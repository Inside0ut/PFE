import unittest2
import cleaner 



class TestCleanningMethods(unittest2.TestCase):

    def test_strip_html(self):
        self.assertEqual(cleaner.strip_html('<p>foo</p>'), 'foo')

    def test_remove_between_braquets(self):
        self.assertEqual(cleaner.remove_between_square_brackets("[foo]"), '')

    def test_denoise_text(self):
        # combines previous functions 
        self.assertEqual(cleaner.denoise_text('<p>[foo]</p>'), '')
    
    def test_remove_special_chars(self):
        self.assertEqual(cleaner.remove_special_characters('foo$#'), 'foo')
    
    def test_stemmer(self):
        self.assertEqual(cleaner.simple_stemmer('eating'), 'eat')
    
    def test_lemmatizer(self):
        self.assertEqual(cleaner.simple_lemmatizer('eater'), 'eat')

    def test_remove_stop_words(self):
        self.assertEqual(cleaner.remove_stopwords('i ate'), 'ate')
    
    def test_expand_contractions(self):
        self.assertEqual(cleaner.expand_contractions("haven't"), 'have not')

if __name__ == '__main__':
    unittest2.main()