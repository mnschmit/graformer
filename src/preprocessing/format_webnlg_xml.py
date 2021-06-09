import argparse
import xml.etree.ElementTree as ET
import json

CATEGORIES = frozenset([
    "Airport",
    "Astronaut",
    "Building",
    "City",
    "ComicsCharacter",
    "Food",
    "Monument",
    "SportsTeam",
    "University",
    "WrittenWork"
])


def main(args):
    num_filtered = 0
    json_dict = {'entries': []}
    for xml_file in args.xml_file:
        xml = ET.parse(xml_file)
        root = xml.getroot()

        for entry in root.iter('entry'):
            if entry.get("category") not in CATEGORIES:
                num_filtered += 1
                continue

            entry_num = len(json_dict['entries']) + 1
            entry_dict = {entry_num: {}}

            entry_dict[entry_num]['modifiedtripleset'] = []
            tripleset = entry.find('modifiedtripleset')
            for triple in tripleset.findall('mtriple'):
                sbj, prd, obj = triple.text.split(' | ')
                entry_dict[entry_num]['modifiedtripleset'].append(
                    {
                        "subject": sbj,
                        "property": prd,
                        "object": obj
                    }
                )

            entry_dict[entry_num]['lexicalisations'] = []
            for lex in entry.findall('lex'):
                entry_dict[entry_num]['lexicalisations'].append(
                    {"lex": lex.text}
                )

            json_dict['entries'].append(entry_dict)

    with open(args.json_file, 'w') as f:
        json.dump(json_dict, f, indent=2)
    print("Number of filtered entries:", num_filtered)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('xml_file', nargs='+')
    parser.add_argument('json_file')
    args = parser.parse_args()
    main(args)
