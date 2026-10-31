"""Preserve reading order for native pages with clearly separated text columns."""

NATIVE_LAYOUT_VERSION = 'columns-v1'


def native_page_text(page):
    blocks = [b for b in page.get_text('blocks') if b[6] == 0 and b[4].strip()]
    # Merge overlapping horizontal extents. Only use column order when the
    # actual blocks leave unambiguous gutters; spanning tables/headings fall
    # back to the existing page order rather than inventing a layout.
    groups = []
    for block in sorted(blocks, key=lambda b: b[0]):
        if groups and block[0] <= groups[-1]['right'] + 2:
            groups[-1]['right'] = max(groups[-1]['right'], block[2])
            groups[-1]['blocks'].append(block)
        else:
            groups.append({'left': block[0], 'right': block[2], 'blocks': [block]})
    columns = 2 <= len(groups) <= 4 and all(
        len(g['blocks']) >= 3
        and max(b[3] for b in g['blocks']) - min(b[1] for b in g['blocks']) >= page.rect.height * .30
        for g in groups)
    gutters = all(groups[i+1]['left'] - g['right'] >= page.rect.width * .015
                  for i, g in enumerate(groups[:-1]))
    if columns and gutters:
        return '\n\n'.join(b[4].strip() for g in groups
                            for b in sorted(g['blocks'], key=lambda b: (b[1], b[0])))
    return page.get_text(sort=True)
