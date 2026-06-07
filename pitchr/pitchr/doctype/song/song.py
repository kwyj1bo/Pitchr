# Copyright (c) 2026, Pratik and contributors
# For license information, please see license.txt

# import frappe
from frappe.model.document import Document


class Song(Document):
	# begin: auto-generated types
	# This code is auto-generated. Do not modify anything in this block.

	from typing import TYPE_CHECKING

	if TYPE_CHECKING:
		from frappe.types import DF

		artist: DF.Data | None
		file_path: DF.Data | None
		melody_contour: DF.LongText | None
		song_name: DF.Data | None
	# end: auto-generated types

	pass
