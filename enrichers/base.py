class Enricher(object):
    name = None
    persistent = True
    requires = tuple()

    def __init__(self, *args, **kwargs):
       pass

    def __call__(self, data):
        raise NotImplementedError

    def add_enrichment(self, data, name, val):
        if 'enrichments' not in data:
            data['enrichments'] = {}
        data['enrichments'][name] = val
        return data

    def has_enrichment(self, data, name):
        return name in data.get('enrichments', {})

    def get_enrichment(self, data, name):
        return data.get('enrichments', {}).get(name)

    def del_enrichment(self, data, name=None):
        enrichment = data.get('enrichments', {})
        if name in enrichment:
            enrichment.pop(name)
        return enrichment

    def get_tokenized_words(self, data):
        for p in data:
            if not (('content' in p) and p['content']):
                continue
            for sent in p['content']:
                yield from (word.lower() for word in sent.split(' ')
                            if len(word) > 1 and not word.isdigit())

    def get_tokenized_sentences(self, data):
        for content in (p['content'] for p in data
                        if 'content' in p and p['content']):
            yield from content

    def get_elements(self, data, tag='body'):
        if self.has_enrichment(data, 'cleaned_' + tag):
            yield from self.get_enrichment(data, 'cleaned_' + tag)
        else:
            yield from (data.get(tag) or [])

            if 'items' in (data.get('_embedded') or {}):
                for item in (data['_embedded']['items'] or []):
                    yield from self.get_elements(item, tag)

    def get_embedded_elements(self, data, tag='body'):
        yield from (data.get('_embedded') or {}).get(tag) or []

        if 'items' in (data.get('_embedded') or {}):
            for item in (data['_embedded']['items'] or []):
                yield from self.get_embedded_elements(item, tag)

    def get_content(self, p):
        if 'content' in p and p['content']:
            return p['content']
        else:
            return ''

    def get_first_content_of_type(self, data, type, tag='body'):
        for p in self.get_elements(data, tag):
            if p.get('type') == type:
                return p.get('content', '')
        return ''

    def content(self, data, tag='body', type_filters=[]):
        yield from (self.get_content(p) for p in self.get_elements(data, tag)
                    if not type_filters or p['type'] in type_filters)

    def paragraph_content(self, data, tag='body'):
        yield from (self.get_content(p) for p in self.get_elements(data, tag)
                    if 'type' in p and p['type'] in ['p', 'lead', 'intro'])

