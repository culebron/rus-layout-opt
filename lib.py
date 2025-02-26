import pandas as pd
import numpy as np
pd.options.display.max_rows = 100
from math import floor
from collections import defaultdict

import colorsys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

# leter categories. V = vowel, C = consonant.
# nb: So-called "soft sign" Ь and "hard sign" Ъ are ancient vowels, and labelled as such.
VOW_CONS_RU = {'v': 'аеёиоуъыьэюя', 'c': 'бвгджзйклмнпрстфхцчшщ'}

# these keys MUST be in any layout, otherwise the code throws an exception and halts
BASE_KEYS_RU = set('ё-!?%*():;йцукенгшщзхъфывапролджэячсмитьбю.,⌴¶')

# abcde fghij = home positions of 0..9 respectively
STANDARD_FINGERS = '''
111123 3678888
001233 66789999
 abcd3 6ghij99
 01233 66789
4
''' 

STANDARD_PENALTIES = '''
753246 6422357
321134 43112357
 10002 2000123
 21114 41112
0
'''

STANDARD_REACH = r'''
753444 5733456
443224 53223456
 11122 2211123
 10014 41001
0
'''

# pos penalty: abs(1 - reach)
# coord penalty:
# 	abs(reach1 - reach2) if same hand
# 	+ 1 if adjacent finger

# hand turn penalty:
# abs of shift1 - shift2
# e.g. col shift @key1 = +2, @key 2 = -1, total = +3 = abs(2 - - 1 = 3)
# @key1 = -2, @key2 = +2, total = 4 = abs(-2 - 2)

# before 21.12.24
# STANDARD_PENALTIES = '''
# 753246 6422246
# 321134 43112357
#  00002 2000023
#  11114 41111
# 0
# '''

# min(standard, 1) for non-home row
# STANDARD_PENALTIES = """
# 642246 6422246
# 422224 43112246
#  00002 2000024
#  11114 41111
# 0
# """

# standard + 2 everywhere in non-home pos
# STANDARD_PENALTIES = """
# 864468 8644468
# 644446 64224468
#  00004 4000046
#  22226 62224
# 0
# """

def map_letter_types(letter, types):
	for letters, tp in types:
		if letter in letters:
			return tp

	return '-'

class Corpus:
	def __init__(self, bigrams):
		self.bigrams = bigrams
		
	def from_string(raw_text, types=None):
		# we take text and encode it, replacing space, linebreak and tab with displayable surrogates.
		# (layouts are encoded with spaces and linebreaks as separators, so this way we won't confuse them)
		text = (raw_text.lower()
			.replace(' ', '⌴')
			.replace('\n', '¶')
			.replace('\t', '→')
			.replace('”', '"')
			.replace('“', '"')
			.replace('«', '"')
			.replace('»', '"')
		)
		nums = defaultdict(int)
		for i in range(2, len(text)):
			nums[text[i-2:i]] += 1

		bigrams = pd.DataFrame(nums.items(), columns=['bigram', 'num'])
		bigrams['l1'] = bigrams.bigram.str[:1]
		bigrams['l2'] = bigrams.bigram.str[1:]

		letter_types = [(v, k) for k, v in (types if types is not None else VOW_CONS_RU).items()]
		for i in (1, 2):
			bigrams[f't{i}'] = bigrams[f'l{i}'].apply(map_letter_types, types=letter_types)
		bigrams['freq'] = bigrams.num / bigrams.num.sum()
		return Corpus(bigrams)
		
	# simple function that reads the corpus and creates a bigram table.
	def from_path(*paths, types=None):
		"""Reads file from path and calculates bigrams frequencies."""
		t = ''
		for path in paths:
			with open(path) as f:
				t += f.read()
		
		return Corpus.from_string(t, types=types)
	
	def display_outerness(self, filter_expr, left_hand=False):
		"""Provide a `filter_expr` to filter the bigrams of the corpus,
		and this function will display a table and a plot with
		where a letter is more often in digrams in the subset.

		E.g. English S more often comes first among consonants.
		So it will be on the right (towards right hand pinky)."""
		d2 = self.bigrams[self.bigrams.eval(filter_expr)]
		t2 = d2.groupby('l1').agg({'freq': 'sum'}).join(d2.groupby('l2').agg({'freq': 'sum'}), how='outer', lsuffix='_out', rsuffix='_in')
		t2.fillna(0, inplace=True)
		t2['outer'] = (t2.freq_out - t2.freq_in) * (-1 if left_hand else 1)
		t2['frequency'] = t2.freq_in + t2.freq_out
		t2['outerness'] = t2.outer / t2.frequency
		t2 *= 10000
		title = 'left hand: pinky <-> index' if left_hand else 'right hand: index <-> pinky'
		ax = t2[['frequency', 'outerness']].plot.scatter(x='outerness', y='frequency', title=title)
		for i, r in t2.iterrows():
			ax.annotate(i, (r.outerness + 200, r.frequency + 5))

		return t2.sort_values('outerness') # to readable numbers

def parse_layer(text):
	"Parses text of a layer of layout, fingers or position penalties."
	keys_map = {}
	for ir, row in enumerate(text.lstrip().rstrip().split('\n')):
		for ic, f in enumerate(row):
			if f != ' ' and f != '∅':
				keys_map[(ir, ic)] = f
	return keys_map

def make_key(row, column, finger: int, penalty: int = 0, reach: int = 0):
	return {
		'row': row,
		'column': column,
		'finger': finger,  # finger unique ID. (left pinky = 0, left ring = 1, ... right pinky = 9)
		'ftype': floor(abs(4.5 - finger)),  # number in its hand (thumb = 0, pinky = 4)
		'hand': (0 if finger < 4.5 else 1), # hand numebre. Left = 0, right = 1
		'penalty': penalty, # position penalty (ie. monogram). From POS_PENALTY
		'reach': reach
	}
	
KEYCAP_LAYER_SHIFTS = {
	0: (0, 0),
	1: (-.2, .2),
	2: (.2, -.2),
	3: (.5, .6),
	-1: (0, 0)
}


def lighten_color(color, amount=0.5):
	c = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(color))
	return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def color_scale(val, min_val, max_val, scale=plt.cm.plasma, lighten=.5):
	return lighten_color(scale((val - min_val) / (max_val - min_val)), lighten)


HOME_POS_NUMS = 'abcdefghij'
class Keyboard:
	"""Keeps fingers and penalties map of a model or a fingers positioning scheme."""
	def __init__(self, name, fingers, penalties, hand_reach=None, key_shape=None, extra_keys=None):
		"""Creates the instance. Fingers and penalties are strings with lines as rows,
		and line positions of chars as columns. They must match exactly.
		
		Parameters
		----------
		- name, str: just the name
		- fingers, str: string, where line is row, char pos is column, and the number in there (0..9)
			is the finger. 0 = left pinky, 1 = left ring, .. 9 = right pinky.
			Letters a..j denote home positions for 0..9 respectively.
			Penalties map and layouts must reproduce these positions.
		- penalties, str: integer penalties in the same positions.
		- hand_reach, str: map, like penalties and fingers, of grades how much you must lower the hand or
			ever lift it low you put the hand
			(or lower it) to reach the key. E.g. to reach key '8' the middle finger must extend to maximum,
			and the hand lie almost flat. The key '<' on QWERTY is very close, so the hand must be risen
			at maximum. To reach top row with index finger, you need to lower the hand at max, but index
			finger can touch the lower row without moving the hand. TODO: this should probably be
			measured with skeleton simulation.
		- key_display, function: a callback that processes a key to be rendered.
			Like adding stagger, depending on row/column.
			Input params: (x: float, y: float, width: float, height: float, keycap: list[str])
			Output: a tuple with the same items.
			This callback won't process `extra_keys`.
		- extra_keys, list: a list of tuples of extra keys to render in the image:
			(x: float, y: float, width: float, height: float, keycap: str).
			Keycap is a single string.

		"""
		self.name = name
		self.extra_keys = extra_keys or []
		self.key_shape = key_shape
		self.keymap = {}
		self.homes = [None] * 10
		finger_count = [0] * 10
		last_pos = [None] * 10
		for (ir, ic), f in parse_layer(fingers).items():
			if f in HOME_POS_NUMS:
				finger = HOME_POS_NUMS.index(f)
				self.homes[finger] = (ir, ic)
				f = finger
				
			f = int(f)
			self.keymap[(ir, ic)] = make_key(ir, ic, f)
			last_pos[f] = (ir, ic)
			finger_count[f] += 1

		for i, pos in enumerate(self.homes):
			if finger_count[i] == 1:
				self.homes[i] = last_pos[i]
			elif finger_count[i] == 0:
				self.homes[i] = (0, 0)
			elif pos is None:
				raise ValueError(f'finger {i} has no home position. Add "{HOME_POS_NUMS[i]}" somewhere.')
		#print('HOMES', self.homes)
		for (ir, ic), p in parse_layer(penalties).items():
			if (ir, ic) not in self.keymap:
				raise ValueError("Penalties map doesn't match fingers map!")
			
			self.keymap[(ir, ic)]['penalty'] = int(p)

		for (ir, ic), r in parse_layer(hand_reach).items():
			if (ir, ic) not in self.keymap:
				raise ValueError("Reach map doesn't match fingers map!")

			self.keymap[(ir, ic)]['reach'] = int(r)

		self.keymap = pd.DataFrame.from_dict(self.keymap, orient='index')
		self.bigrams = {}
		self.monograms = {}
		
		for (r1, c1), key1 in self.keymap.iterrows():
			for (r2, c2), key2 in self.keymap.iterrows():
				rollout = 1 if key1['hand'] == key2['hand'] and key2['ftype'] > key1['ftype'] else 0
				k2penalty = key2['penalty']

				row_cost, row_cat = self.get_row_shift_cost((r1, c1), (r2, c2))
				col_cost, col_cat = self.get_col_shift_cost((r1, c1), (r2, c2))

				self.bigrams[(r1, c1, r2, c2)] = {
					'row_cost': row_cost,
					'row_cat': row_cat,
					'col_cost': col_cost,
					'col_cat': col_cat,
					'rollout': rollout,
					'k2penalty': k2penalty
				}

		self.bigrams = pd.DataFrame.from_dict(self.bigrams, orient='index')
		self.monograms = pd.DataFrame.from_dict(self.monograms, orient='index')
	
	def key_coords(self):
		"""Generates coords data to display the keyboard. 1 unit = 2 cm, step of standard keyboard. Returns:
		`(all_coords, width, height)`, where
		
		* all_coords: a list of tuples for each key: (row, column, x, y, width, height)
		* width: width of the entire keyboard
		* height: height of the entire keyboard
		"""
		all_keys = []
		for (ir, ic) in self.keymap.index:
			if self.key_shape:
				x, y, w, h = self.key_shape(ic, ir, 1, 1)
			else:
				x, y, w, h = ic, ir, 1, 1
			
			all_keys.append((ir, ic, x, y, w, h, None))
		
		width = max(i[2] + i[4] for i in all_keys) - min(i[2] for i in all_keys)		 
		height = max(i[3] + i[5] for i in all_keys) - min(i[3] for i in all_keys) 
		return all_keys, width, height
		
	def get_monogram_cost(self, row, column):
		if (row, column) not in self.keymap:
			return 0
		
		return self.keymap[(row, column)].penalty

	def get_col_shift_cost(self, pos1, pos2):
		k1 = self.keymap.loc[pos1]
		k2 = self.keymap.loc[pos2]

		h1 = self.homes[k1.finger]
		offset1 = h1[1] - pos1[1]
		
		h2 = self.homes[k2.finger]
		offset2 = h2[1] - pos2[1]

		if k1.hand != k2.hand or k1.ftype == 0:
			return offset2 / 2, 'alternating or space'  # if key1 is space or different hand, count only half (half is arbitrary)
		
		return abs(offset2 - offset1), 'ok'

	def get_row_shift_cost(self, pos1, pos2):
		if pos1 not in self.keymap.index:
			return 0, 'key 1 not in kbd'
		k1 = self.keymap.loc[pos1]

		if pos2 not in self.keymap.index:
			return 0, 'key 2 not in kbd'
		k2 = self.keymap.loc[pos2]

		if k1['hand'] != k2['hand']:
			return 0, 'alternating hands'

		if k1['finger'] in (4, 5) or k2['finger'] in (4, 5):
			return 0, 'thumb'

		reach1 = k1['reach']
		reach2 = k2['reach']

		dcol = abs(k2['column'] - k1['column']) + 1
		dreach = abs(reach2 - reach1)

		if k1['finger'] == k2['finger']:
			return (dreach + 2) * 5, 'same finger change rows'  # arbitrary
		return dreach / dcol * 2, ''


	def raw_display(self, key_caps=None, colors=None, title=None):
		all_keys, width, height = self.key_coords()
		for (x, y, w, h, cap) in self.extra_keys:
			all_keys.append((None, None, x, y, w, h, cap))
		
		fig, ax = plt.subplots(1, 1, figsize=(width, height))
		ax.set_axis_off()
		ax.set_title(title)
		
		font = {'family': 'dejavu sans',
			'color':  '#000',
			'size': 14,
			'linespacing': 1.7,
			'ha': 'center'
		}
		
		max_x = max_y = min_x = min_y = 0
		
		for ir, ic, x, y, w, h, cap in all_keys:
			if cap:
				caps = [cap]
			elif key_caps:
				caps = key_caps.get((ir, ic), [])
			else:
				caps = []
			key_color = colors.get((ir, ic), '#ccc')
				
			y *= -1
			min_x = min(x, min_x)
			min_y = min(y, min_y)
			max_x = max(x + w, max_x)
			max_y = max(y + h, max_y)

			# we draw the rectangle making a outer margin of 0.2.
			# We just shift the box right-bottom, and make it narrower and lower by 0.4,
			# so that all keys are still referred by key size (1 unit = 20mm),
			# and this is done consistently everywhere.
			ax.add_patch(Rectangle((x + .2, y + .2 - h + 1), w - .4, h - .4,
			   color=key_color, ec=key_color, 
			   capstyle='round', linewidth=15, linestyle='-', joinstyle='round'))
			
			if key_caps is None:
				continue
			
			if not isinstance(caps, (list, tuple)):
				caps = (caps,)
				
			for layer, cap in enumerate(caps):
				if cap in ('⌴', ' '): continue
				if layer not in KEYCAP_LAYER_SHIFTS:
					raise ValueError(f'too many layers in key caps: {caps} in keymap')

				dx, dy = KEYCAP_LAYER_SHIFTS[layer]
				
				# same shifting for text
				text_y = y + dy * h + - h / 2 + 1
				text_x = x + dx * w + w / 2
				
				font_size = 14 if layer == 0 else 10
				plt.text(text_x, text_y, cap, va='center', fontdict={
					**font, 'color':  '#000', 'size': font_size}) # if key.get('c', 0) else '#444444'})
		
		ax.set_xlim(min_x - .25, max_x + .25)
		ax.set_ylim(min_y - .25, max_y + .25)
		return ax
		
	def display(self):
		"""
		Displays the keyboard. Empty or with `key_caps` from a layout.
		
		"""
		max_pen = self.keymap.penalty.max()
		
		return self.raw_display(
			key_caps=self.keymap['penalty'].to_dict(),
			colors= self.keymap['penalty'].apply(color_scale, args=(0, max_pen)).to_dict(),
			title=f'{self.name} with monogram penalties'
		)

ROW_STAGGER = { 0: 0, 1: .5, 2: .75, 3: 1.25, 4: 1.75 }
				
def std_key_shape(x, y, w, h):
	if x > 6:
		x -= 1

	if x == 0 and y == 1: # '→':
		x -= .5
		w = 1.5
	
	if y == 4: # space bar
		w = 5.25
		x = 2.75
		
	if y == 1 and x == 13: # the / key above the enter
		w = 1.75
	
	if y == 2 and x == 12: # enter
		w = 2.5
		
	if y not in ROW_STAGGER: 
		raise ValueError(f"Row must be 0..=4, got {y} instead.")

	x += ROW_STAGGER[y]

	return x, y, w, h
		
STD_EXTRA_KEYS = [
	(13, 0, 2.25, 1, '← Backspace'),
	
	(0, 2, 1.75, 1, 'Caps Lock'),
	
	(0, 3, 2.25, 1, '↑ Shift'),
	(12.25, 3, 3, 1, '↑ Shift'),

	(0, 4, 1.75, 1, 'Ctrl'),
	(1.75, 4, 1.25, 1, '⊞'),
	(3, 4, 1.5, 1, 'L Alt'),
	(9.75, 4, 1.5, 1, 'AltGr'),
	(11.25, 4, 1.25, 1, '⊞'),
	(12.5, 4, 1.25, 1, '▤↖'),
	(13.75, 4, 1.5, 1, 'Ctrl'),
]
STANDARD_KBD = Keyboard('Standard staggered keyboard', STANDARD_FINGERS, STANDARD_PENALTIES, STANDARD_REACH, std_key_shape, STD_EXTRA_KEYS)


ERGODOX_VSTAG = {2: .1, 3: .2, 4: .1, 10: .1, 11: .2, 12: .1} # x => delta y
def ergodox_key_shape(x, y, w, h):
	if x == 0 and y <= 3:
		w = 1.75
		x -= .75
	
	if x == 14 and y <= 3:
		w = 1.75

	if x in (6, 8) and y in (1, 3):
		h = 1.5
		if y == 3:
			y -= .5

	# thumb blocks
	if y == 5:
		y += .5
		if x in (6, 8):
			y += 1
		else:
			h += 1
	
	# vertical stagger
	if y <= 4:
		y -= ERGODOX_VSTAG.get(x, 0)
	return x, y, w, h


ERGODOX = Keyboard('ergodox',

# note: in the middle columns, there are 2 tall keys, not 3, but I'm not sure how to represent it here,
# so for now, it's set like there are 3.
'''
0012333 6667899
0012333 6667899
0abcd3   6ghij9
0012333 6667899
00123     67899
    e44 55f
''', # ehm... in reality, I press the outermost keys on the top row with the ring fingers, not pinky, so...
# maybe it's better to write the real usage here...

'''
8642468 8642468
3211346 6431123
210002   200012
3211146 6411123
42222     22224
    000 000
''',

'''
7544578 8754457
5322456 6542235
321123   321123
2100124 4210012
00000     00000
    111 111
''',

ergodox_key_shape)



class Layout:
	"""Keeps positions of keys on a particular keyboard."""
	def __init__(self, name, layout_config, debug=False, base_keys=BASE_KEYS_RU):
		"""Initialize the layout. `layout_config` must be either text, or 2-tuple (layout text, Keyboard instance)."""

		if isinstance(layout_config, tuple):
			if len(layout_config) != 2:
				raise ValueError(f'Layout must be either a string, or a 2-tuple (layout, keyboard). Got tuple of {len(layout_config)} instead.')
			layout_text, keyboard = layout_config
		else:
			layout_text, keyboard = (layout_config, STANDARD_KBD)

		layers = layout_text.lstrip().rstrip().split('\n\n')
		maps = [parse_layer(l) for l in layers]
		if debug: print('layout', layers)
		keys = ''.join(k for m in maps for k in m.values())
		if debug: print('layout', keys)
		key_counts = defaultdict(int)
		for k in keys:
			key_counts[k] += 1

		for k in set(keys):
			if key_counts[k] > 1 and k not in ('⌴', '¶', '→'):
				print(f'key "{k}" repeated: {key_counts[k]}')

		missing = base_keys - set(keys)

		if missing:
			raise ValueError(f"Missing keys: {''.join(missing)}, present keys: {''.join(keys)}")

		# making a dict: {letter: (layer, row, column, <finger id>, <finger num in hand>, hand, <pos penalty>)}
		# the last 4 items come from get_finger_props(...) calls
		# make any changes here => change the pd.DataFrame call below
		data = {}
		
		for il, layer in enumerate(maps):
			for (ir, ic), k in layer.items():
				if debug: print(il, ir, ic, k, (ir, ic) in keyboard.keymap.index)
				if k != '∅' and (ir, ic) in keyboard.keymap.index:
					key = keyboard.keymap.loc[(ir, ic)]
					data[k] = {
						'layer': il, 'row': ir, 'column': ic,
						'key_count': key_counts[k],
					}

		self.name = name
		self.keymap = pd.DataFrame.from_dict(data, orient='index')
		self.keyboard = keyboard
		self.original_text = layout_text
		self.base_keys = base_keys
	
	def get_key(self, letter):
		if letter not in self.keymap:
			return None
		r = self.keymap[letter]
		return self.keyboard.keymap[(r['row'], r['column'])]

	def get_pos(self, l):
		if l not in self.keymap.index:
			print('letter "{l}" not in keymap')
			return None

		k = self.keymap.loc[l]
		return k['row'], k['column']

	def get_monogram_cost(self, l2):
		"""Simply looks up keymap and gets pos_penalty field. Lowercases the letters."""
		if l2 not in self.keymap.index:
			if l2.lower() in self.keymap.index:
				l2 = l2.lower() # here we should but don't penalize Shift/AltGr pressing
			else:
				if l2 in self.base_keys or l2.lower() in self.base_keys:
					print(l2)
					print(self.keymap.index)
					raise ValueError(f'base key \'{l2}\' is not in the layout! (may be caused by unquoted backslash)')
				return 0

		row = self.keymap.loc[l2]
		return self.keyboard.get_monogram_cost(row['row'], row['column'])

	# THE MAIN PENALTIES RULES
	# Here we assign costs and also put a text name for the reason why bigram got it,
	# to quickly see WTF is happening

	def get_bigram_cost(self, bg):
		l1, l2 = bg
		pos1 = self.get_pos(l1)
		pos2 = self.get_pos(l2)
		if pos1 is None and pos2 is None:
			return 0, 0, 0, 'l1 and l2 absent'
		if pos1 is None:
			return 0, 0, 0, 'l1 absent'
		if pos2 is None:
			return 0, 0, 0, 'l2 absent'
	
		return self.keyboard.get_col_shift_cost(pos1, pos2), self.keyboard.get_row_shift_cost(pos1, pos2)

	def keycaps(self):
		keycaps = defaultdict(list)
		for k, r in self.keymap.sort_values('layer').iterrows():
			keycaps[(r['row'], r['column'])].append(k)
		return keycaps

	def display(self, what=None):
		"""
		Shows the layout with the keyboard.
		"""
		colors = self.keymap.merge(self.keyboard.keymap, on=['row', 'column']).groupby(['row', 'column']).agg({'finger': 'first'})['finger'].apply(lambda f:
				 lighten_color(plt.cm.Set3((f + (f % 2) * 10) / 20), .5)).to_dict()
		self.keyboard.raw_display(self.keycaps(), colors, f"{self.name} layout with finger zones")

	def export(self):
		cyr = '''
		а a
		б be
		в ve
		г ghe
		д de
		е ie
		ё io
		ж zhe
		з ze
		и i
		й shorti
		к ka
		л el
		м em
		н en
		о o
		п pe
		р er
		с es
		т te
		у u
		ф ef
		х ha
		ц tse
		ч che
		ш sha
		щ shcha
		ъ hardsign
		ы yeru
		ь softsign
		э e
		ю yu
		я ya
		'''

		cyr_other = '''
		ә  schwa 
		і Ukrainian i 
		қ  ka descender
		ң  en descender
		ө  o bar
		ү  u straight
		ұ  u straight_bar
		һ  shha 
		ғ  ghe bar
'''

		LETS = {}
		for c in cyr.strip().split('\n'):
			letter, name = c.strip().split(' ')
			LETS[letter] = f'Cyrillic_{name}'
			LETS[letter.upper()] = f'Cyrillic_{name.upper()}'

		for line in cyr_other.split('\n'):
			line = line[2:]
			if len(line) == 0:
				continue
			letter, prefix, name, postfix = line.split(' ')
			if prefix == '':
				prefix = 'Cyrillic'
			if postfix != '':
				postfix = '_' + postfix
			LETS[letter] = f'{prefix}_{name}{postfix}'
			LETS[letter.upper()] = f'{prefix}_{name.upper()}{postfix}'

		names = r'''
		` backtick
		~ tilde
		! exclam
		@ at
		# numbersign
		№ numerosign
		$ dollar
		% percent
		^ asciicircum
		& ampersand
		* asterisk
		( parenleft
		) parenright
		[ bracketleft
		] bracketright
		{ braceleft
		} braceright
		' apostrophe
		" quotedbl
		, comma
		< less
		. period
		> greater
		/ slash
		? question
		= equal
		+ plus
		\ backslash
		| bar
		- minus 
		_ underscore
		; semicolon
		: colon
		'''

		NAMES_DICT = dict(i.strip().split(' ') for i in names.strip().split('\n'))

		EXCL_POS = {
			(0, 0): 'TLDE',
			(1, 13): 'BKSL',
		}


		rows = ''

		row_names = 'EDCB'
		prev_r = None
		for (r, c), df in self.keymap.sort_values(['row', 'column', 'layer']).groupby(['row', 'column']):
			if r > 3: continue
			if prev_r != r:
				rows += '\n'
			if c > 6: c -= 1
			y = df.reset_index().set_index('layer')
			pos_name = EXCL_POS[(r, c)] if (r, c) in EXCL_POS else f'A{row_names[r]}{c:02d}'	
			
			key_recs = []
			for layer, data in y.iterrows():
				if layer not in y.index: continue
				k = data['index']
				if k in LETS:
					key_recs.append(f"{LETS[k]}, {LETS[k.upper()]}")
				elif data['index'] in NAMES_DICT:
					key_recs.append(NAMES_DICT[k])
				elif k in '1234567890':
					key_recs.append(k)
				
			if len(key_recs) == 0: continue
			rows += f'\tkey <{pos_name}> {{ [ ' + ', '.join(key_recs) + ' ] };\n'
			prev_r = r
		
		print(f'''
default partial alphanumeric_keys
xkb_symbols "{self.name}" {{
	include "ru(common)"
	name[Group1]= "Culebron ({self.name})";
	{rows}
	}};
	''')

class Result:
	# Gets the cost for input KBD text, bigrams & fingers maps
	def __init__(self, corpus, layout):
		bigram_df = corpus.bigrams.copy()
		b = (bigram_df
			.merge(layout.keymap[['row', 'column']], left_on='l1', right_index=True)
			.merge(layout.keymap[['row', 'column']], left_on='l2', right_index=True, suffixes=('1', '2'))
			.merge(layout.keyboard.bigrams, left_on=['row1', 'column1', 'row2', 'column2'], right_index=True, how='left')
			.merge(layout.keyboard.keymap, left_on=['row2', 'column2'], right_index=True)
		)
		b['cost'] = b['num'] * (b['row_cost'] * 3 + b['col_cost'] * 2 + b['k2penalty'] + b['rollout'])

		self.bigrams = b
		self.corpus = corpus # it's not copied here, just a pointer
		self.layout = layout # also not copied
		self.score = b.cost.sum() / b.num.sum()

	def compare(self, other):
		x = self.bigrams[['bigram', 'num', 'row_cat', 'row_cost', 'col_cat', 'col_cost', 'k2penalty', 'rollout', 'cost']].merge(
			other.bigrams[['bigram', 'row_cat', 'row_cost', 'col_cat', 'col_cost', 'k2penalty', 'rollout', 'cost']],
			on='bigram', suffixes=['_old', '_new'])
		x['delta'] = x['cost_new'] - x['cost_old']
		return x[x.delta != 0].sort_values('delta', ascending=False)

	def display(self, *items):
		for what in items:
			show_rows = what == 'rows'
			show_layout = what == 'layout'
			show_costs = what in ('cost', 'costs')
			show_nums = what in ('freq', 'frequencies', 'num', 'nums')
			show_arrows = what in ('arrow', 'arrows')

			if show_rows:
				b = self.bigrams[self.bigrams.l1 != self.bigrams.l2] # exclude same letter pairs
				t1 = b.rename(columns={'l1': 'letter', 'l2': 'other'}).groupby(['letter', 'other']).agg({'num': 'sum'}).reset_index()
				t2 = b.rename(columns={'l2': 'letter', 'l1': 'other'}).groupby(['letter', 'other']).agg({'num': 'sum'}).reset_index()

				stats = pd.concat([t1, t2]).groupby(['letter', 'other']).agg({'num': 'sum'}).reset_index()
				km = self.layout.keymap.merge(self.layout.keyboard.keymap, left_on=['row', 'column'], right_index=True)
				keys = km[['row', 'column', 'hand', 'layer']].reset_index()
				stats2 = stats.merge(keys, left_on='letter', right_on='index')
				stats3 = stats2.merge(keys, left_on='other', right_on='index', suffixes=('', '_other'))
				stats4 = stats3[(stats3.hand == stats3.hand_other) & (stats3.row <= 3) & (stats3.row_other <= 3)].sort_values('layer')
				stats5 = stats4.groupby(['row', 'column', 'row_other']).agg({'letter': 'first', 'num': 'sum'}).reset_index()
				stats6 = stats5.pivot_table('num', ['row', 'row_other'], 'column')
				caps = stats5.groupby(['row', 'column']).agg({'letter': 'first'})['letter'].to_dict()

				stats7 = stats6.melt(ignore_index=False).reset_index().pivot_table('value', ['row', 'column'], 'row_other')
				max_combos = stats7.max().max()

				cc = self.layout.keymap['column']

				fig, axs = plt.subplots(4, cc.max(), figsize=(12, 6))
				fig.suptitle(f'Keys gravitation for {self.layout.name}')
				plt.subplots_adjust(hspace=1)
				for row, axrow in enumerate(axs):
					for col, ax in enumerate(axrow):
						ax.axis('off')
						if (row, col) not in caps:
							continue
						ax.set_xlim(1, max_combos)
						title = caps[(row, col)]
						D = stats7.loc[(row, col)]
						vals = [D[1], D[2], D[3]]
						ax.barh((3, 2, 1), vals)
						ax.set_title(title)
		
			elif show_layout:
				self.layout.display()
			elif show_costs:
				df = self.bigrams.groupby(['row', 'column']).agg({'cost': 'sum', 'num': 'sum'})
				df['meancost'] = df['cost'] / df['num']

				min_cost = df['meancost'].min()
				max_cost = df['meancost'].max()

				colors = df['meancost'].apply(color_scale, args=(min_cost, max_cost)).to_dict()
				self.layout.keyboard.raw_display(
					df['meancost'].round(2).to_dict(), colors, f'{self.layout.name} costs (on 2nd keys of bigrams)')

			elif show_nums:
				total = self.bigrams['num'].sum()

				gr = self.bigrams.groupby(['row', 'column']).agg({'num': 'sum', 'l2': 'first'})
				signs = gr.apply(lambda r: f'{r["l2"]}\n{r["num"]}', axis=1)
				nums = gr['num']

				min_num = nums.min()
				max_num = nums.max()
				colors = nums.apply(color_scale, args=(min_num, max_num, plt.cm.viridis))
				self.layout.keyboard.raw_display(
					signs.to_dict(), colors.to_dict(), f'{self.layout.name} frequencies (on 2nd keys of bigrams)')

			elif show_arrows:
				_, width, height = self.layout.keyboard.key_coords()
				fig, ax = plt.subplots(1, 1, figsize=(width, height))
				self.show_arrows(ax)

			else:
				raise ValueError('what must be `cost` or `freq`.')

	def show_arrows(self, ax=None, max_num=None, costs=None):
		letters = self.bigrams[self.bigrams.l1 != '⌴'][['column1', 'row1', 'num', 'cost']].groupby(['column1', 'row1']).agg({'num': 'sum', 'cost': 'sum'})
		maxnum = letters['num'].max()
		maxcost = (letters['cost'] / letters['num']).max() ** .5
		
		pairs = self.bigrams
		km = self.layout.keymap.reset_index().merge(self.layout.keyboard.keymap, on=['row', 'column']).set_index('index')
		km2 = km.reset_index().set_index(['layer', 'row', 'column'])
		x1 = pairs['l1'].map(km['column'])
		y1 = pairs['l1'].map(km['row'])
		pairs = pairs.merge(
			self.layout.keyboard.keymap[['hand', 'finger']], left_on=['row1', 'column1'], right_index=True, suffixes=('', '1')
			).merge(
			self.layout.keyboard.keymap[['hand', 'finger']], left_on=['row2', 'column2'], right_index=True, suffixes=('', '2')
		)

		num_threshold = 200
		pairs2 = pairs[(pairs.hand1 == pairs.hand2) & (pairs.l2 != '¶')
			& ~pairs.finger1.isin([4, 5]) & ~pairs.finger2.isin([4, 5])
			& (pairs.num > num_threshold)].copy()

		# pairs2.to_csv(f'/tmp/{}_pairs2.csv')

		all_coords, width, height = self.layout.keyboard.key_coords()

		ax.set_xlim((0, width))
		ax.set_ylim((0, height))
		ax.set_axis_off()
		ax.set_title(f'most popular bigrams in {self.layout.name}')

		coords = {}

		minx = min(i[2] for i in all_coords)
		miny = min(i[3] for i in all_coords)

		for ir, ic, x, y, w, h, _ in all_coords:
			if (ic, ir) not in letters.index:
				continue
			row = letters.loc[(ic, ir)]
			color = color_scale((row['cost'] / row['num']) ** .5, 0, maxcost, plt.cm.rainbow)
			n = (row['num'] / maxnum) * .5
			X = x - minx + 1
			Y = height + miny - y - .5

			if (0, ir, ic) in km2.index:
				cap = km2.loc[(0, ir, ic)]['index']
				if cap[0] == '⌴':
					continue
				ax.text(X, Y, cap[0].replace('⌴', ''),
					fontdict={'color':  '#000', 'size': 14, 'ha': 'center'})

			ax.add_patch(FancyBboxPatch((X - n * w / 2, Y - h * n / 2), n * w, n * h,
			capstyle='round', linewidth=0, color=color))
			coords[(ic, ir)] = np.array([X, Y])
			
		# import ipdb; ipdb.set_trace()
		cc = pairs2['cost'] / pairs2['num'] #pairs2['bigram_cost'] = pairs2['coord_cost'] + pairs2['move_cost']
		max_num = max(pairs2['num'].max() ** .5, max_num or 0)
		min_cost = cc.min() ** .5
		max_cost = cc.max() ** .5
		if costs is not None:
			min_cost = min(min_cost, costs[0])
			max_cost = max(max_cost, costs[1])

		for i, bg in pairs2.sort_values('cost').iterrows():
			l1, l2 = bg['l1'], bg['l2']
			if l1 not in km.index or l2 not in km.index:
					continue
			r1, c1 = km.loc[l1][['row', 'column']]
			r2, c2 = km.loc[l2][['row', 'column']]
			if (c1, r1) not in coords or (c2, r2) not in coords:
					continue

			coords1 = coords[(c1, r1)]
			coords2 = coords[(c2, r2)]
			delta = coords2 - coords1
			
			# print(f'min cost {min_cost}, max cost {max_cost}, bg cost {bg["cost"]} num {bg["num"]}, price {bg["cost"] / bg["num"]}')
			t = ((bg['cost'] / bg['num']) ** .5 - min_cost) / (max_cost - min_cost)
			ax.arrow(coords1[0], coords1[1], delta[0], delta[1],
				width=bg['num'] ** .5 / max_num / 4,
				shape='left', length_includes_head=True, ec='#00000000',
				color=plt.cm.turbo(t))

	def combodata(self):
		filtered = self.bigrams[(self.bigrams.l2 != '⌴') & (self.bigrams.l1 != '⌴')].copy()
		l1 = filtered.rename(columns={'l1': 'l'})[['l', 'cost', 'num']]
		l2 = filtered.rename(columns={'l2': 'l'})[['l', 'cost', 'num']]
		ll = pd.concat([l1, l2]).groupby('l').agg({'cost': 'sum', 'num': 'sum'})
		ll = ll.merge(self.layout.keymap[['layer', 'row', 'column']], left_index=True, right_index=True
				 ).reset_index(names='letter').groupby(['layer', 'row', 'column']
				 ).agg({'cost': 'sum', 'num': 'sum', 'letter': 'first'})

		ll['price'] = ll.cost / ll.num
		return ll

	def combomap(self, *others):
		layouts = [(i, i.layout.keyboard.key_coords()) for i in (self, *others)]
		width = max(i[1][1] for i in layouts)
		all_heights = sum(i[1][2] for i in layouts) + 1
		fig, axes = plt.subplots(len(layouts), 1, figsize=(width, all_heights))

		ll = pd.concat([i[0].bigrams for i in layouts])
		ll = ll[ll['num'] > 0].copy()

		#ll.to_csv('/tmp/maxcosts.csv')
		min_cost = (ll['cost'] / ll['num']).min() ** .5
		max_cost = (ll['cost'] / ll['num']).max() ** .5
		max_num = ll['num'].max() ** .5

		plt.text(width / 2, all_heights / len(layouts) + .5, 'Layouts comparison.\nArrow size = frequency, color = total cost.\nKey size = its bigrams cost, color = mean price.\nScales are the same.',
			 size=14, ha='center')
		for (result, (all_coords, width, height)), ax in zip(layouts, axes):
			result.show_arrows(ax, max_num, (min_cost, max_cost))

	def combochart(self, other):
		self_ll = self.combodata()
		other_ll = other.combodata()
		ll2 = self_ll.merge(other_ll, on='letter', how='outer', suffixes=['1', '2'])
		ax = ll2.plot.scatter(x='num1', y='cost1', figsize=(10, 6), s=50)
		for i, r in ll2.iterrows():
			ax.annotate(r['letter'].replace('¶', '\\n'), (r['num1'] + 1000, r['cost1']))
		ll2.plot.scatter(x='num2', y='cost2', ax=ax, color='#c44')
		for i, r in ll2.iterrows():
			ax.annotate(r['letter'].replace('¶', '\\n'), (r['num2'] + 1000, r['cost2']))
		ax.set_title(f'Comparison of {self.layout.name} (blue) and {other.layout.name} (red)')

	def load_bars(self, ax=None):
		d = self.bigrams
		d = d[d.finger != 4].groupby('finger').agg({'num': 'sum'})
		d.plot.bar(title=self.layout.name, legend=False, ax=ax)

def compare(results_dict, key1, key2):
	return results_dict[key1].compare(results_dict[key2])

