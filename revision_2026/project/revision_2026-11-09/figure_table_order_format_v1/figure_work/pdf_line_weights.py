"""Adjust stroke-only PDF paths in place, preserving clipping and paint order."""
from io import BytesIO
import math

import fitz
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, FloatObject


def product(left, right):
    a, b, c, d = left
    e, f, g, h = right
    return (a * e + b * g, a * f + b * h,
            c * e + d * g, c * f + d * h)


def minimum_scale(matrix):
    a, b, c, d = matrix
    square_sum = a * a + b * b + c * c + d * d
    determinant = a * d - b * c
    discriminant = max(0, square_sum * square_sum - 4 * determinant * determinant)
    return math.sqrt(max(0, (square_sum - math.sqrt(discriminant)) / 2))


def adjust_stroke_widths(document, minimum_width, color_replacements=()):
    reader = PdfReader(BytesIO(document.tobytes()))
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    page = writer.pages[0]
    forms = {}

    def collect(stream, resources, initial=(1, 0, 0, 1), initial_width=1):
        matrix, width, stack = initial, initial_width, []
        for operands, operator in ContentStream(stream, writer).operations:
            if operator == b'q':
                stack.append((matrix, width))
            elif operator == b'Q':
                matrix, width = stack.pop()
            elif operator == b'cm':
                matrix = product(tuple(float(x) for x in operands[:4]), matrix)
            elif operator == b'w':
                width = float(operands[0])
            elif operator == b'Do':
                obj = resources.get('/XObject', {}).get(operands[0])
                if obj is None:
                    continue
                obj = obj.get_object()
                if obj.get('/Subtype') != '/Form':
                    continue
                form_matrix = tuple(float(x) for x in obj.get('/Matrix', [1, 0, 0, 1, 0, 0])[:4])
                effective = product(form_matrix, matrix)
                key = obj.indirect_reference.idnum
                if key not in forms or minimum_scale(effective) < minimum_scale(forms[key][2]) - 1e-10:
                    res = obj.get('/Resources', resources)
                    forms[key] = (obj, res, effective, width)
                    collect(obj, res, effective, width)

    collect(page.get_contents(), page['/Resources'])
    count = 0

    def rewrite(stream, initial=(1, 0, 0, 1), initial_width=1):
        nonlocal count
        content = ContentStream(stream, writer)
        matrix, width, stack = initial, initial_width, []
        operations = []
        for operands, operator in content.operations:
            if operator in (b'rg', b'RG') and len(operands) == 3:
                rgb = tuple(float(x) for x in operands)
                for original, replacement in color_replacements:
                    if all(abs(a - b) < 0.0001 for a, b in zip(rgb, original)):
                        operands = [FloatObject(value) for value in replacement]
                        break
            if operator == b'q':
                stack.append((matrix, width))
            elif operator == b'Q':
                matrix, width = stack.pop()
            elif operator == b'cm':
                matrix = product(tuple(float(x) for x in operands[:4]), matrix)
            elif operator == b'w':
                width = float(operands[0])
            scale = minimum_scale(matrix)
            if operator in (b'S', b's') and scale > 0 and width * scale < minimum_width - 0.001:
                operations.append(([FloatObject(minimum_width / scale)], b'w'))
                operations.append((operands, operator))
                operations.append(([FloatObject(width)], b'w'))
                count += 1
            else:
                operations.append((operands, operator))
        content.operations = operations
        return content

    page.replace_contents(rewrite(page.get_contents()))
    for obj, resources, matrix, width in forms.values():
        changed = rewrite(obj, matrix, width)
        obj.set_data(changed.get_data())
    buffer = BytesIO()
    writer.write(buffer)
    return fitz.open(stream=buffer.getvalue(), filetype='pdf'), count
