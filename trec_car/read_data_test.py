from .read_data import *
import sys

if len(sys.argv)<4:
    print("usage ",sys.argv[0]," articlefile outlinefile paragraphfile")
    exit()

articles=sys.argv[1]
outlines=sys.argv[2]
paragraphs=sys.argv[3]


# to open either pages or outlines use iter_annotations

with open(articles, 'rb') as f:
    for p in iter_pages(f):
        print('\npagename:', p.page_name)
        print('\npageid:', p.page_id)
        print('\nmeta:', p.page_meta)

        # get one data structure with nested (heading, [children]) pairs
        headings = p.nested_headings()
        #print(headings)

        if len(p.outline())>0:
            print( p.outline()[0].__str__())

            print('deep headings= ', [ (str(section.heading), len(children)) for (section, children) in p.deep_headings_list()])

            print('flat headings= ' ,["/".join([str(section.heading) for section in sectionpath]) for sectionpath in p.flat_headings_list()])




with open(outlines, 'rb') as f:
    for p in iter_annotations(f):
        print('\npagename:', p.page_name)

        # get one data structure with nested (heading, [children]) pairs
        headings = p.nested_headings()
        print('headings= ',  [ (str(section.heading), len(children)) for (section, children) in headings])

        if len(p.outline())>2:
            print('heading 1=', p.outline()[0])

            print('deep headings= ', [ (str(section.heading), len(children)) for (section, children) in p.deep_headings_list()])

            print('flat headings= ' ,["/".join([str(section.heading) for section in sectionpath]) for sectionpath in p.flat_headings_list()])

# exit(0)


with open(paragraphs, 'rb') as f:
    for p in iter_paragraphs(f):
        print('\n', p.para_id, ':')

        # Print just the text
        texts = [elem.text if isinstance(elem, ParaText)
                 else elem.anchor_text
                 for elem in p.bodies]
        print(' '.join(texts))

        # Print just the linked entities
        entities = [elem.page
                    for elem in p.bodies
                    if isinstance(elem, ParaLink)]
        print(entities)

        # Print text interspersed with links as pairs (text, link)
        mixed = [(elem.anchor_text, elem.page) if isinstance(elem, ParaLink)
                 else (elem.text, None)
                 for elem in p.bodies]
        print(mixed)

