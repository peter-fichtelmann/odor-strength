from bs4 import BeautifulSoup
import re
from .molecules.html_getter import get_static_html

class GoodScentsOnePageParser:
    def extract_descriptions_from_container(self, match_pattern, soup):
        descriptions = []
        # match_pattern = "(Odor Description)|(Odor:)"
        for match in soup.find_all(string=re.compile(match_pattern)):
            try:
                # parent = str(match.parent)
                parent = match.parent.get_text()
                # print(str(match.parent.parent))
                #(Odor Description:|Odor:)
                description = re.search(rf"{match_pattern}(.*)", parent).group(0)
                description = description.replace('<span>','').replace('</span>','').replace('</div>','').replace('<div>','').replace('</td>', '')
                description = re.sub(rf'{match_pattern}\s?', '', description)
                description = re.sub('&amp;', '&', description) # reverse & encoding
                description = re.sub(r' sourceofdescriptionArctander.*', '', description, flags=re.DOTALL)
                # print(description)
                # text = []
                # for span in parent.find_all("span"):
                #     text.append(span.get_text())
                # description = ''.join(text)
                description = description.replace('\n','').replace('\t','').replace('\r','')
                descriptions.append(description.lower())
            except:
                continue
        return descriptions
    
    def extract_physical_properties(self, soup):
        physical_properties = {}

        section = soup.find(string="Physical Properties:")
        if section:
            try:
                container = section.parent.next_sibling.next_sibling
                # print(container.find_all("tr"))
                for row in container.find_all("tr"):
                    key_tag = row.find(class_="radw4")
                    value_tag = row.find(class_="radw11")
                    if key_tag and value_tag:  # make sure both exist
                        key = key_tag.get_text(strip=True)
                        value = value_tag.get_text(strip=True)
                        i = 2
                        original_key = key
                        while key in physical_properties.keys():
                            key = original_key + f"_{i}"
                            i += 1
                        physical_properties[key] = value
            except:
                pass
        return physical_properties 
    
    def extract_cas(self, soup, url):
        cas = soup.find('td', class_='radw11')
        cas = cas.text.strip() if cas else None
        if len(soup.find_all(string='CAS Number:')) > 1:
            print(f'More than 1 CAS Number in {url}')
        return cas

    def href_to_cas(self, href):
        if href:
            href = 'http://www.thegoodscentscompany.com/' + href
            html = get_static_html(href)
            cas = self.extract_cas(BeautifulSoup(html, 'html.parser'), href)
        else:
            cas = None
        return cas
    
    def get_impurity_data(self, tr):
        a_tag = tr.find('a') if tr else None
        href = a_tag['href'] if a_tag else None
        name = a_tag.get_text() if a_tag else None
        concentration = tr.find('td', class_='radw7') if tr else None
        concentration = concentration.get_text() if concentration else None
        if name is not None and concentration is not None:
            concentration = concentration.replace(name, '')
        else:
            concentration = None
        cas = self.href_to_cas(href)
        return {'name': name, 'concentration': concentration, 'href': href, 'cas': cas}

    def get_impurities(self, soup):
        tr_contain = soup.find(string=re.compile(r'Also(\(can\))? Contains:'))
        tr_contain = tr_contain.parent.parent if tr_contain else None
        impurity_1 = self.get_impurity_data(tr_contain)
        next_tr = tr_contain.find_next_sibling('tr') if tr_contain else None
        impurity_2 = self.get_impurity_data(next_tr)
        return {'impurity_1': impurity_1, 'impurity_2': impurity_2}
    
    def get_odor_description_with_source_detail(self, soup):
        source_details = []
        names2 = []
        odor_descriptions = []
        for odor_description in soup.find_all(string=re.compile(r'(Odor Description:|Odour Description:|Odor:.*|Odour:.*)')):
            source_detail = ''
            name2 = ''
            if 'Odor Description:' in odor_description or 'Odour Description:' in odor_description:
                odor_desc_td = odor_description.parent
                odor_description = odor_desc_td.get_text() if odor_desc_td else ''
                prev_tr = odor_desc_td.find_parent('tr')
                prev_tr = prev_tr.find_previous_sibling('tr') if prev_tr else None
                prev_td_radw5 = prev_tr.find('td', class_='radw5') if prev_tr else None
                if prev_td_radw5:
                    additional_descriptor = prev_td_radw5.get_text()  # add link descriptors because not always equivalent to odor description
                    odor_description = additional_descriptor + ' ' + odor_description
                    if 'Luebke' in odor_desc_td.get_text() or 'Mosciano' in odor_desc_td.get_text():
                        source_detail = odor_desc_td.get_text().split('sourceofdescription')[1]
                else:
                    prev_td_radw6 = prev_tr.find('td', class_='radw6') if prev_tr else None
                    if prev_td_radw6:
                        source_detail = prev_td_radw6.get_text()
                    else:
                        prev_prev_tr = prev_tr.find_previous_sibling('tr') if prev_tr else None
                        prev_prev_td_radw6 = prev_prev_tr.find('td', class_='radw6') if prev_prev_tr else None
                        if prev_prev_td_radw6:
                            source_detail = prev_prev_td_radw6.get_text()
                            name2 = prev_tr.find('td').get_text()
                        # else:
                        #     print(f'source detail/name2 not found {url}')
            else:
                td = odor_description.parent
                if td.find('a'):
                    name2 = td.find('a').get_text()
                prev_tr = td.find_parent('tr')
                prev_tr = prev_tr.find_previous_sibling('tr') if prev_tr else None
                source_detail = prev_tr.find('td').get_text() if prev_tr else ''
                if source_detail == 'For experimental / research use only.':
                    prev_prev_tr = prev_tr.find_previous_sibling('tr') if prev_tr else None
                    source_detail = prev_prev_tr.get_text() if prev_prev_tr else ''
            odor_descriptions.append(odor_description)
            source_details.append(source_detail)
            names2.append(name2)
        return odor_descriptions, source_details, names2 

    def get_soup(self, url): 
        info_dict = {'url':url, 'useful':True}
        html = get_static_html(url)
        # html = get_html(url, save_path)
        if 'Odor Strength' not in str(html) and 'Odor Description:' not in str(html) and 'Odor:' not in str(html): # no taste description but odor instead of original
            info_dict['useful'] = False
            return None, info_dict
        soup = BeautifulSoup(html, "html.parser")
        return soup, info_dict

    def parse_one_page(self, url):
        soup, info_dict = self.get_soup(url)
        if soup is None:
            return info_dict
        for br in soup.find_all('br'):
            # br.replace_with(" " + br.text)
            # # Get the parent of the <br> tag and remove all siblings after the <br>
            # next_elements = list(br.find_all_next())
            # for element in next_elements:
            #     element.decompose()  # Remove subsequent elements
            br.replace_with(" sourceofdescription" + br.text)  # Remove the <br> tag itself
        try:
            name = soup.find(attrs={"itemprop":"name"}).get_text()
            info_dict.update({'name':name})
        except:
            pass
        try:
            cas = self.extract_cas(soup, url)
            info_dict.update({'cas':cas})
            # smiles = self.smiles_converter.cas_to_isomeric_smiles(cas)
            # info_dict.update({'raw_goodscents_smiles':smiles})
        except:
            pass
        # odor_descriptions = self.extract_descriptions_from_container(r'(Odor Description:|Odor:)', soup)
        odor_strength = self.extract_descriptions_from_container(r'(Odor Strength:)', soup)
        substantivity = self.extract_descriptions_from_container(r'(Substantivity:)', soup)
        concentration = self.extract_descriptions_from_container(r'(Recommendation for)', soup)
        impurities = self.get_impurities(soup)
        source_detail = {}
        names2 = {} 
        physical_properties = self.extract_physical_properties(soup)
        odor_descriptions, source_detail, names2 = self.get_odor_description_with_source_detail(soup)
        # if len(odor_descriptions) == 0 or odor_descriptions == []:
        #     info_dict.update({'useful':False})
        # else:
        #     print(get_odor_description_with_source_detail(soup))
        #     odor_descriptions = list(set(odor_descriptions)) # remove duplicates
        #     tds_radw5 = soup.find_all('td', class_='radw5')
        #     tds_product = soup.find_all('td', itemtype="https://schema.org/Product")
        #     for i, odor_description in enumerate(odor_descriptions):
        #         for td in tds_radw5:
        #             if odor_description == re.sub(r' sourceofdescriptionarctander.*', '', td.get_text().lower().replace('odor description:', ''), flags=re.DOTALL) and 'Odor Description' in td.get_text():
        #                 odor_desc_td = td
        #                 prev_tr = odor_desc_td.find_parent('tr')
        #                 prev_tr = prev_tr.find_previous_sibling('tr') if prev_tr else None
        #                 prev_td_radw5 = prev_tr.find('td', class_='radw5') if prev_tr else None
        #                 if prev_td_radw5:
        #                     additional_descriptor = prev_td_radw5.get_text()  # add link descriptors because not always equivalent to odor description
        #                     odor_descriptions[i] = additional_descriptor + ' ' + odor_description
        #                     if 'Luebke' in td.get_text() or 'Mosciano' in td.get_text():
        #                         source_detail[odor_descriptions[i]] = td.get_text().split('sourceofdescription')[1]
        #                         names2[odor_descriptions[i]] = ''
        #                 else:
        #                     # print(prev_tr)
        #                     prev_td_radw6 = prev_tr.find('td', class_='radw6') if prev_tr else None
        #                     if prev_td_radw6:
        #                         source_detail[odor_description] = prev_td_radw6.get_text()
        #                         names2[odor_description] = ''
        #                     else:
        #                         prev_prev_tr = prev_tr.find_previous_sibling('tr') if prev_tr else None
        #                         prev_prev_td_radw6 = prev_prev_tr.find('td', class_='radw6') if prev_prev_tr else None
        #                         if prev_prev_td_radw6:
        #                             source_detail[odor_description] = prev_prev_td_radw6.get_text()
        #                             names2[odor_description] = prev_tr.find('td').get_text()
        #                         else:
        #                             print(f'source detail/name2 not found {url}')
        #                             source_detail[odor_description] = ''
        #                             names2[odor_description] = ''
        #         for td in tds_product:
        #             if len(td.find_all(class_='wrd15')) > 0:
        #                 if odor_description == re.sub(r'.*?odor:\s?', '', td.find(class_='wrd15').get_text().lower(), flags=re.DOTALL) and 'Odor:' in td.get_text():
        #                     if td.find('a'):
        #                         names2[odor_description] = td.find('a').get_text()
        #                     else:
        #                         names2[odor_description] = ''
        #                     prev_tr = td.find_parent('tr')
        #                     prev_tr = prev_tr.find_previous_sibling('tr') if prev_tr else None
        #                     source_detail[odor_description] = prev_tr.find('td').get_text() if prev_tr else ''
        #         if odor_description not in source_detail:
        #             source_detail[odor_description] = ''
        #         if odor_description not in names2:
        #             names2[odor_description] = ''
        if len(odor_descriptions) != len(source_detail) and len(odor_descriptions) != len(names2):
            print(f'WARNING: Mismatch of odor descriptions {len(odor_descriptions)} and source details {len(source_detail)} and names2 {len(names2)}')
        # odor_data = "\t\t".join(odor_descriptions)
        info_dict.update({'raw_description':odor_descriptions, 'source_detail':source_detail, 'name2': names2, 'impurities': impurities})
        if odor_strength:
            info_dict.update({'odor_strength':odor_strength[0]})
        if substantivity:
            info_dict.update({'substantivity':substantivity[0]})
        if concentration:
            all_text_list = list(map(str.lower, soup.find_all(string=True)))
            concentration_i = all_text_list.index('recommendation for ' + concentration[0])
            concentration = soup.find_all(string=True)[concentration_i+2]
            concentration = re.search(r'\d+\.\d+\s+%', concentration)
            concentration = concentration.group() if concentration else None
            info_dict.update({'concentration':concentration})
        info_dict.update(physical_properties)
        return info_dict