from collections import OrderedDict
import enrichers
from enrichers.base import Enricher


class Pipeline(Enricher):
    name = 'pipeline'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        enricher_names = args[0]
        satisfied = set()
        self.enrichers = OrderedDict()
        for e in enricher_names:
            enrich_class = enrichers.get_enricher(e)
            self.enrichers[e] = enrich_class(*args, **kwargs)
            satisfied.add(e)

    def __call__(self, data):
        for e in self.enrichers.values():
            data = e(data)
        for e in self.enrichers.values():
            if not e.persistent:
                e.del_enrichment(data, name=e.name)
        return data

    def get_enricher(self, enricher_names):
        return self.enrichers.get(enricher_names)
