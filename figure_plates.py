"""Assemble related supplementary diagnostics into vector evidence plates."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
from string import ascii_lowercase
import subprocess
from xml.etree import ElementTree as ET


SUPPLEMENT_PLATES = {
    'actor_controls': (('fixed_transcripts', 2), ('cremad', 3)),
    'score_controls': (('ordinal_controls', 1), ('conditional_intensity', 2)),
    'frozen_targets': (('moud_transfer', 1), ('frozen_dialogue', 1), ('msp_face_transfer', 1)),
    'refit_checks': (('msp_face_refit', 2),),
    'encoder_checks': (('visual_components', 2),),
    'gap_checks': (('gap_location', 4),),
    'observation_checks': (('dispersion_pooling', 2), ('face_crops', 2), ('audio_channel', 2)),
    'ensemble_checks': (('seed_ensemble', 3),),
    'probability_checks': (('neutral_readout', 2), ('probability_pool', 1)),
    'view_checks': (('view_stability', 2),),
    'weighting_checks': (('class_weighting', 2),),
    'modality_checks': (('audiovisual_example', 2),),
    'intensity_signal_checks': (('modality_agreement', 1), ('seed_sensitivity', 2), ('frozen_lnln_heads', 2)),
    'calibration_checks': (('marginal_tilt', 2), ('intensity_calibration', 2)),
}

SVG = 'http://www.w3.org/2000/svg'
ET.register_namespace('', SVG)
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')


@dataclass(frozen=True)
class PlatePart:
    path: Path
    letters: str


def plate_parts(directory, blocks):
    offset = 0
    for name, panel_count in blocks:
        yield PlatePart(directory / f'{name}.pdf', ascii_lowercase[offset:offset + panel_count])
        offset += panel_count


def namespace_ids(root, prefix):
    ids = {node.attrib['id']: prefix + node.attrib['id'] for node in root.iter() if 'id' in node.attrib}
    for node in root.iter():
        for key, value in node.attrib.items():
            if key == 'id':
                node.set(key, ids[value])
            elif value.startswith('#') and value[1:] in ids:
                node.set(key, '#' + ids[value[1:]])
            elif 'url(#' in value:
                node.set(key, re.sub(r'url\(#([^)]*)\)', lambda x: f'url(#{ids[x[1]]})', value))


def assemble_plate(output, parts):
    """Write a single vector PDF and SVG, with matching print-resolution PNG."""
    directory = parts[0].path.parent
    inset = 5.
    sources = []
    for part in parts:
        original = ET.parse(part.path.with_suffix('.svg')).getroot()
        _, _, native_width, native_height = map(float, original.attrib['viewBox'].split())
        sources.append((part, original, native_width, native_height))
    width = min(485., max(item[2] for item in sources) + 2 * inset)
    rows = []
    y = 3.
    for part, original, native_width, native_height in sources:
        scale = min(1., (width - 2 * inset) / native_width)
        w, h = native_width * scale, native_height * scale
        rows.append((part, original, (width - w) / 2, y, w, h, scale))
        y += h + 7.
    height = y - 4.

    svg = ET.Element(f'{{{SVG}}}svg', width=f'{width}pt', height=f'{height}pt',
                     viewBox=f'0 0 {width} {height}')
    ET.SubElement(svg, f'{{{SVG}}}rect', width=str(width), height=str(height), fill='white')
    tex = []
    for index, (part, original, x, top, w, h, scale) in enumerate(rows):
        namespace_ids(original, f'p{index}_')
        original.attrib.update(x=str(x), y=str(top), width=str(w), height=str(h))
        svg.append(original)
        tex.append(r'\put(%.5f,%.5f){\includegraphics[width=%.5fbp]{%s}}'
                   % (x, height - top - h, w, part.path.name))
        placements = json.loads(part.path.with_suffix('.panels.json').read_text(encoding='utf-8'))
        for row in placements:
            letters = row['letters']
            baseline = top + scale * row['baseline_from_top']
            for letter, offset in zip(letters, row['anchors_from_left'], strict=True):
                anchor = x + scale * offset
                label = ET.SubElement(svg, f'{{{SVG}}}text', x=f'{anchor:.5f}', y=f'{baseline:.5f}',
                                      fill='black', attrib={'font-family': 'Arial, sans-serif',
                                                            'font-size': '9', 'font-weight': 'bold'})
                label.text = letter
                tex.append(r'\put(%.5f,%.5f){\sffamily\bfseries\fontsize{9bp}{10bp}\selectfont %s}'
                           % (anchor, height - baseline, letter))

    name = output.stem
    source = directory / f'{name}.tex'
    source.write_text('\n'.join([
        r'\documentclass{article}', r'\usepackage{graphicx,helvet}',
        r'\pagestyle{empty}', r'\begin{document}',
        rf'\paperwidth={width}bp', rf'\paperheight={height}bp',
        r'\pdfpagewidth=\paperwidth', r'\pdfpageheight=\paperheight',
        r'\pdfhorigin=0bp', r'\pdfvorigin=0bp',
        r'\setlength{\unitlength}{1bp}', rf'\setbox0=\hbox{{\begin{{picture}}({width},{height})',
        *tex, r'\end{picture}}', r'\shipout\vbox{\offinterlineskip\box0}', r'\end{document}',
    ]) + '\n', encoding='utf8')
    subprocess.run(['pdflatex', '-interaction=batchmode', '-halt-on-error', source.name], cwd=directory,
                   check=True, stdout=subprocess.DEVNULL)
    output.write_bytes(source.with_suffix('.pdf').read_bytes())
    subprocess.run(['pdftoppm', '-r', '360', '-png', '-singlefile', str(output),
                    str(output.with_suffix(''))], check=True)
    ET.ElementTree(svg).write(output.with_suffix('.svg'), encoding='utf-8', xml_declaration=True)
