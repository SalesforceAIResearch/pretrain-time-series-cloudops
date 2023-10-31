from gluonts.dataset import DataEntry
from gluonts.dataset.common import ProcessStartField, ProcessDataEntry
from gluonts.transform import SimpleTransformation


class ProcessDataEntryTransform(SimpleTransformation):
    def __init__(self, freq: str, univariate: bool):
        self.process = ProcessDataEntry(
            freq=freq, one_dim_target=univariate, use_timestamp=False
        )

    def transform(self, data: DataEntry) -> DataEntry:
        return self.process(data)
