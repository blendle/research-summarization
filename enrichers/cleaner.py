import regex
from bs4 import BeautifulSoup
from enrichers.base import Enricher


class Cleaner(Enricher):
    name = 'cleaner'
    cleaned_tags = ['body']
    persistent = False
    requires = ()

    def __call__(self, data):
        body = self.get_elements(data)
        clean_body = [self.clean_content(part) for part in body]

        return self.add_enrichment(data, 'cleaned_body', clean_body)

    def del_enrichment(self, data, name=None):
        enrichment = data.get('enrichments', {})
        for t in self.cleaned_tags:
            name = 'cleaned_' + t
            if name in enrichment:
                enrichment.pop(name)
        return enrichment

    def clean_content(self, part):
        if self.get_content(part):
            # Remove html tags
            cleaned_content = BeautifulSoup(self.get_content(part), 'html.parser').text
            # Replace '…' with '...'
            cleaned_content = regex.sub('…', '...', cleaned_content)
            # Replace funny single quotes with a normal single quote (')
            cleaned_content = regex.sub('[`‘’‛⸂⸃⸌⸍⸜⸝]', "'", cleaned_content)
            # Replace funny double quotes (including double single quotes ('')
            # and double commas) with a normal double quote (")
            cleaned_content = regex.sub("[\p{Pi}\p{Pf}„“]|('')|(,,)", '"', cleaned_content)

            part['content'] = cleaned_content

        return part
