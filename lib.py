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

STANDARD_FINGERS = '''
001233 6678999
001233 66789999
 01233 6678999
 01233 66789
4
''' 

STANDARD_PENALTIES = '''
753246 6422246
321134 43112357
 00002 2000023
 11114 41111
0
'''

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

class Corpus:
	def __init__(self, bigrams):
		self.bigrams = bigrams
		
	def from_string(raw_text):
		# we take text and encode it, replacing space, linebreak and tab with displayable surrogates.
		# (layouts are encoded with spaces and linebreaks as separators, so this way we won't confuse them)
		text = raw_text.lower().replace(' ', '⌴').replace('\n', '¶').replace('\t', '→')

		nums = defaultdict(int)
		for i in range(2, len(text)):
			nums[text[i-2:i]] += 1

		bigrams = pd.DataFrame(nums.items(), columns=['bigram', 'num'])
		bigrams['l1'] = bigrams.bigram.str[:1]
		bigrams['l2'] = bigrams.bigram.str[1:]
		for i in (1, 2):
			bigrams[f't{i}'] = bigrams[f'l{i}'].map(lambda l: 'v' if l.lower() in VOW_CONS_RU['v'] else ('c' if l.lower() in VOW_CONS_RU['c'] else '-'))
		bigrams['freq'] = bigrams.num / bigrams.num.sum()
		return Corpus(bigrams)
		
	# simple function that reads the corpus and creates a bigram table.
	def from_path(path):
		"""Reads file from path and calculates bigrams frequencies."""
		with open(path) as f:
			return Corpus.from_string(f.read())
	
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


def get_finger_props(finger):
	return {
		'finger': finger,  # finger unique ID. (left pinky = 0, left ring = 1, ... right pinky = 9)
		'ftype': floor(abs(4.5 - finger)),  # number in its hand (thumb = 0, pinky = 4)
		'hand': (0 if finger < 4.5 else 1), # hand numebre. Left = 0, right = 1
		'penalty': 0, # position penalty (ie. monogram). From POS_PENALTY
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


class Keyboard:
	"""Keeps fingers and penalties map of a model or a fingers positioning scheme."""
	def __init__(self, name, fingers, penalties, key_shape=None, extra_keys=None):
		"""Creates the instance. Fingers and penalties are strings with lines as rows,
		and line positions of chars as columns. They must match exactly.
		
		Parameters
		----------
		- name, str: just the name
		- fingers, str: string, where line is row, char pos is column, and the number in there (0..9)
			is the finger. 0 = left pinky, 1 = left ring, .. 9 = right pinky.
			Penalties map and layouts must reproduce these positions.
		- penalties, str: integer penalties in the same positions.
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
		for (ir, ic), f in parse_layer(fingers).items():
			props = get_finger_props(int(f))
			self.keymap[(ir, ic)] = props

		for (ir, ic), p in parse_layer(penalties).items():
			if (ir, ic) not in self.keymap:
				raise ValueError("Penalties map doesn't match fingers map!")
			
			self.keymap[(ir, ic)]['penalty'] = int(p)
	
	def key_coords(self):
		"""Generates coords data to display the keyboard. 1 unit = 2 cm, step of standard keyboard. Returns:
		`(all_coords, width, height)`, where
		
		* all_coords: a list of tuples for each key: (row, column, x, y, width, height)
		* width: width of the entire keyboard
		* height: height of the entire keyboard
		"""
		all_keys = []
		for (ir, ic), k in self.keymap.items():
			if self.key_shape:
				x, y, w, h = self.key_shape(ic, ir, 1, 1)
			else:
				x, y, w, h = ic, ir, 1, 1
			
			all_keys.append((ir, ic, x, y, w, h, None))
		
		width = max(i[2] + i[4] for i in all_keys) - min(i[2] for i in all_keys)		 
		height = max(i[3] + i[5] for i in all_keys) - min(i[3] for i in all_keys) 
		return all_keys, width, height
		
		
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
				text_y = y + dy * h + - h / 2 + 1 - .1
				text_x = x + dx * w + w / 2
				
				font_size = 14 if layer == 0 else 10
				plt.text(text_x, text_y, cap, fontdict={
					**font, 'color':  '#000', 'size': font_size}) # if key.get('c', 0) else '#444444'})
		
		ax.set_xlim(min_x - .25, max_x + .25)
		ax.set_ylim(min_y - .25, max_y + .25)
		return ax
		
	def display(self):
		"""
		Displays the keyboard. Empty or with `key_caps` from a layout.
		
		"""
		max_pen = max(k['penalty'] for k in self.keymap.values())
		
		return self.raw_display(
			key_caps={coord: [k['penalty']] for coord, k in self.keymap.items()},
			colors={coord: color_scale(k['penalty'], 0, max_pen) for coord, k in self.keymap.items()},
			title = f'{self.name} with monogram penalties'
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
STANDARD_KBD = Keyboard('Standard staggered keyboard', STANDARD_FINGERS, STANDARD_PENALTIES, std_key_shape, STD_EXTRA_KEYS)


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
001233   667899
0012333 6667899
00123     67899
    444 555
''', # ehm... in reality, I press the outermost keys on the top row with the ring fingers, not pinky, so...
# maybe it's better to write the real usage here...

'''
8642468 8642468
2111346 6431112
200002   200002
2111146 6411112
42222     22224
    000 000
''', ergodox_key_shape)



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
				if debug: print(il, ir, ic, k, (ir, ic) in keyboard.keymap)
				if k != '∅' and (ir, ic) in keyboard.keymap:
					data[k] = {'layer': il, 'row': ir, 'column': ic,
							   'key_count': key_counts[k],
							   **keyboard.keymap[(ir, ic)]}

		self.name = name
		self.keymap = pd.DataFrame.from_dict(data, orient='index')
		self.keyboard = keyboard
		self.original_text = layout_text
		self.base_keys = base_keys


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

		return self.keymap.loc[l2].penalty


	# THE MAIN PENALTIES RULES
	# Here we assign costs and also put a text name for the reason why bigram got it,
	# to quickly see WTF is happening
	def get_bigram_cost(self, bigram):
		l1, l2 = bigram

		if l2 not in self.keymap.index:
			if l2.lower() in self.keymap.index:
				l2 = l2.lower() # lowercase (= no penalties for shifts)
			else:
				return 0, 'L2 not in kbd'
		k2 = self.keymap.loc[l2]

		if l1 not in self.keymap.index:
			if l1.lower() in self.keymap.index:
				l1 = l1.lower() # lowercase of l1.
			else:
				return 0, 'L1 not in kbd'
		k1 = self.keymap.loc[l1]

		rules = (
			(k1.ftype == 0 or k2.ftype == 0, 0, 'space bar'),
			(k1.hand != k2.hand, 0, 'altern hands'),
			(l1 == l2, 0, 'same key'),

			(abs(k2.row - k1.row) >= 2 and k1.ftype == k2.ftype, 8, 'same finger over row'),
			(k1.ftype == k2.ftype, 6, 'same finger adj row'),

			(k1.ftype == 1 and k2.ftype == 4, 2, 'index -> pinky'),
			(k1.ftype == 3 and k2.ftype == 4 and abs(k1.row - k2.row) == 1, 5, 'ring -> pinky, next row'),
			(k1.ftype == 4 and k2.ftype == 3 and abs(k1.row - k2.row) == 1, 3, 'pinky -> ring, next row'),

			(abs(k1.ftype - k2.ftype) == 1 and abs(k2.row - k1.row) > 1, 10, 'adj finger over row'),
			(abs(k1.ftype - k2.ftype) == 2 and abs(k2.row - k1.row) > 1, 8, 'over 1 finger, over 1 row'),
			(k1.ftype == 4 and k2.ftype == 1 and abs(k2.row - k1.row) > 1, 4, 'pinky -> index over 1 row'),
			(k1.ftype == 1 and k2.ftype == 4 and abs(k2.row - k1.row) > 1, 6, 'over 2 fingers, over 1 row'),
			
			(k1.ftype > k2.ftype + 1 and k2.row == k1.row, 0, 'in, over 1 finger, same row'),
			(k1.ftype > k2.ftype + 1 and abs(k2.row - k1.row) == 1, 1, 'in, over 1 finger, adj row'),
			(k1.ftype == k2.ftype + 1 and k2.row <= k1.row, 2, 'in, adj finger, same or adj row'),
			(k1.ftype > k2.ftype and k2.row > k1.row, 1, 'in, lower row'),
			
			(k1.ftype == 1 and k2.ftype == 2 and k1.row == k2.row, 1, 'index->middle same row'),
			(k2.ftype > k1.ftype, 4, 'out, over one finger'),
			(k1.ftype + 1 == k2.ftype and k1.row == k2.row, 3, 'out, next finger'),
			(k1.ftype + 1 == k2.ftype and abs(k1.row - k2.row) >= 1, 5, 'out, next finger'),
		)

		for cond, penalty, reason in rules:
			if cond:
				return penalty, reason

		return 4, 'none'

	def keycaps(self):
		keycaps = defaultdict(list)
		for k, r in self.keymap.sort_values('layer').iterrows():
			keycaps[(r['row'], r['column'])].append(k)
		return keycaps

	def display(self):
		"""
		Shows the layout with the keyboard.
		"""
		
		colors = self.keymap.groupby(['row', 'column']).agg({'finger': 'first'})['finger'].apply(lambda f:
				 lighten_color(plt.cm.Set3((f + (f % 2) * 10) / 20), .5)).to_dict()
		self.keyboard.raw_display(self.keycaps(), colors, f"{self.name} layout with finger zones")


class Result:
	# Gets the cost for input KBD text, bigrams & fingers maps
	def __init__(self, corpus, layout):
		bigram_df = corpus.bigrams.copy()

		# taking the text of keyboard layout and encode it into keymap a dataframe
		bigram_df['price_l2'] = bigram_df.l2.apply(layout.get_monogram_cost)

		# calculate bigrams cost
		bigram_df[['price_di', 'category']] = bigram_df.bigram.apply(lambda d: pd.Series(layout.get_bigram_cost(d)))
		bigram_df['price'] = bigram_df.price_l2 + bigram_df.price_di
		bigram_df['cost'] = bigram_df.price * bigram_df.num
		bigram_df['finger'] = bigram_df['l2'].map(layout.keymap.finger)
		bigram_df['column'] = bigram_df['l2'].map(layout.keymap.column)
		bigram_df['row'] = bigram_df['l2'].map(layout.keymap.row)
		self.bigrams = bigram_df
		self.corpus = corpus # it's not copied here, just a pointer
		self.layout = layout # also not copied
		self.score = bigram_df.cost.sum() / bigram_df.num.sum()

	def compare(self, other):
		x = self.bigrams[['bigram', 'num', 'category', 'price', 'cost']].merge(
			other.bigrams[['bigram', 'category', 'price', 'cost']],
			on='bigram', suffixes=['_old', '_new'])
		x['delta'] = x['cost_new'] - x['cost_old']
		return x[x.delta != 0].sort_values('delta', ascending=False)

	def display(self, *items):
		for what in items:
			show_layout = what == 'layout'
			show_costs = what in ('cost', 'costs')
			show_nums = what in ('freq', 'frequencies', 'num', 'nums')
			show_arrows = what in ('arrow', 'arrows')

			if show_layout:
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

				nums = self.bigrams.groupby(['row', 'column']).agg({'num': 'sum'})['num']

				min_num = nums.min()
				max_num = nums.max()
				colors = nums.apply(color_scale, args=(min_num, max_num, plt.cm.viridis))
				self.layout.keyboard.raw_display(
					np.log(nums).round(1).to_dict(), colors.to_dict(), f'{self.layout.name} frequencies (on 2nd keys of bigrams)')

			elif show_arrows:
				_, width, height = self.layout.keyboard.key_coords()
				fig, ax = plt.subplots(1, 1, figsize=(width, height))
				self.show_arrows(ax)

			else:
				raise ValueError('what must be \`cost\` or \`freq\`.')

	def show_arrows(self, ax=None, max_num=None, costs=None):
		letters = self.bigrams[['column', 'row', 'num', 'cost']].groupby(['column', 'row']).agg({'num': 'sum', 'cost': 'sum'})
		maxnum = letters['num'].max()
		maxcost = (letters['cost'] / letters['num']).max()

		pairs = self.bigrams
		km = self.layout.keymap
		km2 = km.reset_index().set_index(['layer', 'row', 'column'])
		x1 = pairs['l1'].map(km['column'])
		y1 = pairs['l1'].map(km['row'])
		pairs['h1'] = pairs['l1'].map(km['hand'])
		pairs['h2'] = pairs['l2'].map(km['hand'])
		num_threshold = 200
		pairs2 = pairs[(pairs.h1 == pairs.h2) & (pairs.l2 != '¶')
			   & (pairs.l1 != '⌴') & (pairs.l2 != '⌴')
			   & (pairs.num > num_threshold) & (pairs.price > 1.5)]

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
			color = color_scale(row['cost'] / row['num'], 0, maxcost, plt.cm.rainbow)
			n = (row['num'] / maxnum)
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


		max_num = max(pairs2['num'].max() ** .5, max_num or 0)
		min_cost = pairs2['cost'].min() ** .5
		max_cost = pairs2['cost'].max() ** .5
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
			
			t = (bg['cost'] ** .5 - min_cost) / (max_cost - min_cost)
			ax.arrow(coords1[0], coords1[1], delta[0], delta[1],
				width=bg['num'] ** .5 / max_num / 5,
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


def compare(results_dict, key1, key2):
	return results_dict[key1].compare(results_dict[key2])
