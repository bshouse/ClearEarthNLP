#!/usr/bin/python3

import codecs
import requests
from bs4 import BeautifulSoup


def run_parser(input_text):
	url = 'http://verbs.colorado.edu/~ribh9977/cgi-bin/run_parser.py'
	files = {'text': input_text, 'dropdown': 'Text'}
	#files = {'text': codecs.open(input_file, encoding='utf-8'),'dropdown': 'Text'}
	r = requests.post(url, files=files)
	html = r.text.encode('ascii','ignore')
	output = r.text #NOTE html
	return output

def run_onto(input_text):
	pass

def run_ner(input_text):
	pass

def run_srl(input_text):
	pass
