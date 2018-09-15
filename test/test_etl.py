import unittest
import json
import etl


class TestETL(unittest.TestCase):

    def test_header(self):
        line = """
        {
    "relation": [
        [
            "Date",
            "Sep 21",
            "Sep 28",
            "Oct 5",
            "Oct 12",
            "Oct 19",
            "Oct 26",
            "Nov 2",
            "Nov 9",
            "Nov 16",
            "Nov 23"
        ],
        [
            "Opponent",
            "at San Diego",
            "Brown",
            "at Holy Cross",
            "at Cornell",
            "Lafayette",
            "Princeton",
            "Dartmouth",
            "at Columbia",
            "Penn",
            "at Yale"
        ],
        [
            "Score",
            "W, 42-20",
            "W, 41-23",
            "W, 41-35",
            "W, 34-24",
            "W, 35-16",
            "L, 51-48",
            "W, 24-21",
            "W, 34-0",
            "W, 38-30",
            "W, 34-7"
        ],
        [
            "pts",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "-",
            "0",
            "0"
        ],
        [
            "rush",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-"
        ],
        [
            "rec",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-"
        ],
        [
            "kr",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-"
        ],
        [
            "pr",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-"
        ],
        [
            "int",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-"
        ],
        [
            "fum",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-"
        ],
        [
            "xpm",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-"
        ],
        [
            "fgm",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-"
        ],
        [
            "saf",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-"
        ],
        [
            "misc",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-"
        ],
        [
            "2pt",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-"
        ]
    ],
    "pageTitle": "Austin Taylor - Harvard",
    "title": "",
    "url": "http://www.gocrimson.com/sports/fball/2013-14/bios/taylor_austin_k3j6?view=gamelog&pos=qb",
    "hasHeader": true,
    "headerPosition": "FIRST_ROW",
    "tableType": "RELATION",
    "tableNum": 10,
    "s3Link": "common-crawl/crawl-data/CC-MAIN-2015-32/segments/1438044271733.81/warc/CC-MAIN-20150728004431-00342-ip-10-236-191-2.ec2.internal.warc.gz",
    "recordEndOffset": 463152139,
    "recordOffset": 463128794,
    "tableOrientation": "HORIZONTAL",
    "TableContextTimeStampAfterTable": "{222436=\u00a9 Copyright 2014 Harvard University All Rights Reserved}",
    "textBeforeTable": "Austin Taylor 97 97 Related Bios Kyle Adams Sean Ahern Eric Baars Sam Batiste Jordan Becerra Andrew Berg David Bicknell Zach Boden Joshua Boyd Blade Brady Cameron Brate Ben Braunecker Cory Briggs Kolbi Brown Matt Brown Blaine Burgess Nick Burrello Darien Carr Andrew Casten Tyler Caveness Donovan Celerin Dayne Davis Ryan Delisle Dominick DeLucia Jack Dittmer James Duberg Nick Easton Chris Evans Andrew Ezekoye Anthony Fabiano Ben Falloon Anthony Firkser Andrew Fischer Andrew Flesher Joseph Foster Justin Fox Danny Frate David Gawlas Asante Gibson Ryan Halvorson Tyler Hamblin Norman Hayes Conner Hempel Zach Hodges Jason Holdway Scott Hosch Caleb Johnson Casey Johnson Ryan Jones Paul Kaczor Reynaldo Kirton Matt Koran Andrew Larson Adam Ledford Christian Lee David Leopard Jacob Lindsey Connor Loftus Colton Lynch Michael Mancinelli Matt Martindale James Martter Miles McCollum Raishaun McGhee Jameson McShea Eric Medes Dan Melow Jimmy Meyer Scott Miller Dexter Monroe Dan Moody",
    "textAfterTable": "Date Opponent Score comp att pct yds y/a td int sac yds Sep 21 at San Diego W, 42-20 - - - - - - - - - Sep 28 Brown W, 41-23 - - - - - - - - - Oct 5 at Holy Cross W, 41-35 - - - - - - - - - Oct 12 at Cornell W, 34-24 - - - - - - - - - Oct 19 Lafayette W, 35-16 - - - - - - - - - Oct 26 Princeton L, 51-48 - - - - - - - - - Nov 2 Dartmouth W, 24-21 - - - - - - - - - Nov 9 at Columbia W, 34-0 - - - - - - - - - Nov 16 Penn W, 38-30 - - - - - - - - - Nov 23 at Yale",
    "hasKeyColumn": true,
    "keyColumnIndex": 0,
    "headerRowIndex": 0
}
        """
        data = json.loads(line)
        table = etl.Table(data)
        self.assertEqual(table.get_header()[1], 'Opponent')


if __name__ == '__main__':
    unittest.main()
