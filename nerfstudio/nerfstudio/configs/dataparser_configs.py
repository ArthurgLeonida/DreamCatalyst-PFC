# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dataparser configs — trimmed for DreamCatalyst-NS.
Only nerfstudio and colmap dataparsers are kept.
"""

from typing import TYPE_CHECKING

import tyro

from nerfstudio.data.dataparsers.base_dataparser import DataParserConfig
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig

dataparsers = {
    "nerfstudio-data": NerfstudioDataParserConfig(),
    "colmap": ColmapDataParserConfig(),
}

all_dataparsers = {**dataparsers}

if TYPE_CHECKING:
    DataParserUnion = DataParserConfig
else:
    DataParserUnion = tyro.extras.subcommand_type_from_defaults(
        all_dataparsers,
        prefix_names=False,
    )

AnnotatedDataParserUnion = tyro.conf.OmitSubcommandPrefixes[DataParserUnion]
"""Union over possible dataparser types, annotated with metadata for tyro."""
