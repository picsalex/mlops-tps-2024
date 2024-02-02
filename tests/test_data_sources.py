from src.steps.data.datalake_initializers import data_source_list_initializer


class TestDataSources:
    def test_data_source_initializer(self):
        data_source_list = data_source_list_initializer()

        assert len(data_source_list.data_sources) > 0
        assert len(data_source_list.data_sources[0].label_map) > 0
        