import unittest
import pymupdf
from app.services.pdf_layout import native_page_text
from app.services.document_processor import extract_pages
from unittest.mock import patch


class PDFLayoutTests(unittest.TestCase):
    def test_columns_are_not_interleaved_by_height(self):
        with pymupdf.open() as doc:
            page = doc.new_page(width=600, height=700)
            for x, name in [(30, 'LEFT'), (230, 'MIDDLE'), (430, 'RIGHT')]:
                for y in (50, 250, 450):
                    page.insert_text((x, y), f'{name} section {y}')
            text = native_page_text(page)
            self.assertLess(text.index('LEFT section 450'), text.index('MIDDLE section 50'))
            self.assertLess(text.index('MIDDLE section 450'), text.index('RIGHT section 50'))
            self.assertEqual(text.count('section'), 9)

    def test_single_column_and_blank_page_keep_default_text(self):
        with pymupdf.open() as doc:
            page = doc.new_page()
            self.assertEqual(native_page_text(page), '')
            page.insert_text((30, 50), 'One ordinary paragraph.')
            self.assertEqual(native_page_text(page), page.get_text(sort=True))

    def test_old_native_cache_is_rebuilt_without_repeating_vision_ocr(self):
        with pymupdf.open() as doc:
            page=doc.new_page()
            page.insert_text((30,50),'Fresh native text with corrected ordering.')
            saved={'1':{'page':1,'text':'Old mixed columns','method':'native','status':'ok'}}
            with patch('app.services.document_processor.needs_ocr',return_value=False), \
                 patch('app.services.document_processor.transcribe_image') as ocr:
                result=extract_pages(doc.tobytes(),'notes.pdf',cached=saved)
            self.assertIn('Fresh native text',result[0]['text'])
            self.assertEqual(result[0]['layout_version'],'columns-v1')
            ocr.assert_not_called()
