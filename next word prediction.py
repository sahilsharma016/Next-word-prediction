#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
import numpy as np
import tensorflow as tf


# In[2]:


data='''

 This article is about the war ongoing since 2014. For the escalation since 2022, see Russian invasion of Ukraine. For other wars between the two nations, see List of wars between Russia and Ukraine.
Russo-Ukrainian War
Part of the post-Soviet conflicts






Clockwise from top left:
Ukrainian tanks during the 2022 Kharkiv counteroffensive;

Russian-installed officials in Moscow ratifying the annexation of four Ukrainian regions;

Russian proxy forces during the Donbas war;

Russian bombing during the Siege of Mariupol;

Russian soldiers during the invasion of Crimea;

Civilians killed by Russian missile strikes on Kyiv
Date	20 February 2014[d] – present
(9 years, 4 months, 1 week and 5 days)
Location	
Ukraine, Russia, and Black Sea (spillover into Poland, Moldova and Belarus)
Status	Ongoing
Territorial
changes	
Russian annexation of Crimea and parts of four southeast Ukrainian oblasts in 2014 and 2022, respectively
Russian occupation of about 18% of Ukrainian territory as of November 2022[1]
 
Belligerents
 Ukraine
Supplied by:
For countries providing aid to Ukraine since 2022, see foreign aid to Ukraine	
 Russia

 Donetsk PR[a] (2014–2022)
 Luhansk PR[b] (2014–2022)
Supported by:

 Belarus[c] (2022–present)

Supplied by:
For details, see Russian military suppliers
Commanders and leaders
 Ukraine

Volodymyr Zelenskyy
(2019–present)
Petro Poroshenko
(2014–2019)
Oleksandr Turchynov
(acting; 2014)
Oleksii Reznikov
(2021–present)
Andriy Taran
(2020–2021)
Andrii Zahorodniuk
(2019–2020)
Stepan Poltorak
(2014–2019)
Valeriy Heletey
(2014)
Mykhailo Koval
(2014)
Ihor Tenyukh
(2014)
Valerii Zaluzhnyi
(2021–present)
Ruslan Khomchak
(2019–2021)
Arsen Avakov
(2014–2021)
 Russia

Vladimir Putin
Sergei Shoigu
Valery Gerasimov
Yevgeny Prigozhin
Sergey Aksyonov
Aleksey Chaly
(2014)
Denis Pushilin
Pavel Gubarev
(2014)
Igor Girkin
(2014)
Leonid Pasechnik
(2017–present)
Igor Plotnitsky
(2014–2017)
Valery Bolotov
(2014)
 Belarus

Alexander Lukashenko
Strength
For details of strengths and units involved at key points in the conflict, see:
Combatants of the war in Donbas (2014–2022)
Order of battle for the Russian invasion of Ukraine
Casualties and losses
Reports vary widely, but tens of thousands at a minimum.[2][3] See Casualties of the Russo-Ukrainian War for details.
vte
Russo-Ukrainian War (outline)
vte
Post-Soviet conflicts
The Russo-Ukrainian War,[e] previously referred to as the Ukrainian crisis in its early stages,[4] is an ongoing international conflict between Russia, alongside Russian-backed separatists, and Ukraine, which began in February 2014.[f] Following Ukraine's Revolution of Dignity, Russia annexed Crimea from Ukraine and supported pro-Russian separatists fighting the Ukrainian military in the Donbas war. The first eight years of conflict also included naval incidents, cyberwarfare, and heightened political tensions. In February 2022, Russia launched a full-scale invasion of Ukraine.

In early 2014, the Euromaidan protests led to the Revolution of Dignity and the ousting of Ukraine's pro-Russian president Viktor Yanukovych. Shortly after, pro-Russian unrest erupted in eastern and southern Ukraine. Simultaneously, unmarked Russian troops moved into Ukraine's Crimea and took over government buildings, strategic sites and infrastructure. Russia soon annexed Crimea after a highly-disputed referendum. In April 2014, armed pro-Russian separatists seized government buildings in Ukraine's eastern Donbas region and proclaimed the Donetsk People's Republic (DPR) and Luhansk People's Republic (LPR) as independent states, starting the Donbas war. The separatists received considerable but covert support from Russia, and Ukrainian attempts to fully retake separatist-held areas failed. Although Russia denied involvement, Russian troops took part in the fighting. In February 2015, Russia and Ukraine signed the Minsk II agreements to end the conflict, but the agreements were never fully implemented in the years that followed. The Donbas war settled into a violent but static conflict between Ukraine and Russian proxies, with many brief ceasefires but no lasting peace and few changes in territorial control.

Beginning in 2021, Russia built up a large military presence near its border with Ukraine, including within neighbouring Belarus. Russian officials repeatedly denied plans to attack Ukraine. Russian president Vladimir Putin criticized the enlargement of NATO and demanded that Ukraine be barred from ever joining the military alliance. He also expressed irredentist views and questioned Ukraine's right to exist. Russia recognized the DPR and LPR as independent states in February 2022, with Putin announcing a "special military operation" in Ukraine and subsequently invading the region. The invasion was internationally condemned; many countries imposed sanctions against Russia and increased existing sanctions. Russia abandoned an attempt to take Kyiv in early April 2022 amid fierce resistance. From August, Ukrainian forces began recapturing territories in the north-east and south as a result of counter-offensives. In late September, Russia declared the annexation of four partially-occupied regions in southern and eastern Ukraine, which was internationally unrecognized. It spent the winter conducting failed offensive operations in the Donbas, and in the spring dug into positions for an anticipated Ukrainian counteroffensive. The war has resulted in a refugee crisis and tens of thousands of deaths.
Background
Main article: Russia–Ukraine relations
See also: Historical background of the 2014 pro-Russian unrest in Ukraine
Independent Ukraine and the Orange Revolution
Further information: Orange Revolution
After the dissolution of the Soviet Union (USSR) in 1991, Ukraine and Russia maintained close ties. In 1994, Ukraine agreed to accede to the Treaty on the Non-Proliferation of Nuclear Weapons as a non-nuclear-weapon state.[5] Former Soviet nuclear weapons in Ukraine were removed and dismantled.[6] In return, Russia, the United Kingdom, and the United States agreed to uphold the territorial integrity and political independence of Ukraine through the Budapest Memorandum on Security Assurances.[7][8] In 1999, Russia was one of the signatories of the Charter for European Security, which "reaffirmed the inherent right of each and every participating State to be free to choose or change its security arrangements, including treaties of alliance, as they evolve."[9] In the years after the dissolution of the USSR, several former Eastern Bloc countries joined NATO, partly in response to regional security threats involving Russia such as the 1993 Russian constitutional crisis, the War in Abkhazia (1992–1993) and the First Chechen War (1994–1996). Putin claimed Western powers broke promises not to let any Eastern European countries join.[10][11]


Protesters in Independence Square in Kyiv during the Orange Revolution, November 2004
The 2004 Ukrainian presidential election was controversial. During the election campaign, opposition candidate Viktor Yushchenko was poisoned by TCDD dioxin;[12][13] he later accused Russia of involvement.[14] In November, Prime Minister Viktor Yanukovych was declared the winner, despite allegations of vote-rigging by election observers.[15] During a two-month period which became known as the Orange Revolution, large peaceful protests successfully challenged the outcome. After the Supreme Court of Ukraine annulled the initial result due to widespread electoral fraud, a second round re-run was held, bringing to power Yushchenko as president and Yulia Tymoshenko as prime minister, and leaving Yanukovych in opposition.[16] The Orange Revolution is often grouped together with other early-21st century protest movements, particularly within the former USSR, known as colour revolutions. According to Anthony Cordesman, Russian military officers viewed such colour revolutions as an attempt by the US and European states to destabilise neighbouring countries and undermine Russia's national security.[17] Russian President Vladimir Putin accused organisers of the 2011–2013 Russian protests of being former advisors to Yushchenko, and described the protests as an attempt to transfer the Orange Revolution to Russia.[18] Rallies in favour of Putin during this period were called "anti-Orange protests".[19]

At the 2008 Bucharest summit, Ukraine and Georgia sought to join NATO. The response among NATO members was divided; Western European countries opposed offering Membership Action Plans (MAP) in order to avoid antagonising Russia, while US President George W. Bush pushed for their admission.[20] NATO ultimately refused to offer Ukraine and Georgia MAPs, but also issued a statement agreeing that "these countries will become members of NATO" at some point. Putin voiced strong opposition to Georgia and Ukraine's NATO membership bids.[21] By January 2022, the possibility of Ukraine joining NATO remained remote.[22]

In 2009, Yanukovych announced his intent to again run for president in the 2010 Ukrainian presidential election,[23] which he subsequently won.[24] In November 2013, a wave of large, pro-European Union (EU) protests erupted in response to Yanukovych's sudden decision not to sign the EU–Ukraine Association Agreement, instead choosing closer ties to Russia and the Eurasian Economic Union. On 22 February 2013 the Ukrainian parliament had overwhelmingly approved of finalizing the agreement with the EU,[25] subsequent to which Russia had put pressure on Ukraine to reject it.[26]

Euromaidan, Revolution of Dignity, and pro-Russian unrest
Main articles: Euromaidan, Revolution of Dignity, and 2014 pro-Russian unrest in Ukraine
Following months of protests as part of the Euromaidan movement, on 21 February 2014 Yanukovych and the leaders of the parliamentary opposition signed a settlement agreement that called for early elections. The following day, Yanukovych fled from the capital ahead of an impeachment vote that stripped him of his powers as president.[27][28][29][30] On 23 February, the parliament adopted a bill to repeal the 2012 law which gave Russian language an official status.[31] The bill was not enacted,[32] however, the proposal provoked negative reactions in the Russian-speaking regions of Ukraine,[33] intensified by Russian media saying that the ethnic Russian population was in imminent danger.[34]

On 27 February, an interim government was established and early presidential elections were scheduled. The following day, Yanukovych resurfaced in Russia and in a press conference declared that he remained the acting president of Ukraine, just as Russia was beginning its overt military campaign in Crimea. Leaders of Russian-speaking eastern regions of Ukraine declared continuing loyalty to Yanukovych,[28][35] causing the 2014 pro-Russian unrest in Ukraine.

Russian military bases in Crimea
Main article: Political status of Crimea
At the onset of the conflict, Russia had roughly 12,000 military personnel in the Black Sea Fleet,[34] in several locations in the Crimean peninsula like Sevastopol, Kacha, Hvardiiske, Simferopol Raion, Sarych, and others. In 2005 a dispute broke out over control of the Sarych cape lighthouse near Yalta, and a number of other beacons.[36][37] Russian presence was allowed by the basing and transit agreement with Ukraine. Under the agreements the Russian military in Crimea was constrained to a maximum of 25,000 troops; they were required to: respect the sovereignty of Ukraine, honor its legislation, not interfere in the internal affairs of the country, and show their "military identification cards" when crossing the international border.[38] Early in the conflict, the agreement's sizeable troop limit allowed Russia to significantly reinforce its military presence under the plausible guise of security concerns, deploy special forces and other required capabilities to conduct the operation in Crimea.[34]

According to the original treaty on the division of the Soviet Black Sea Fleet signed in 1997, Russia was allowed to have its military bases in Crimea until 2017, after which it would evacuate all military units including its portion of the Black Sea Fleet out of the Autonomous Republic of Crimea and Sevastopol. On 21 April 2010, former Ukrainian president Viktor Yanukovych signed a new deal known as the Kharkiv Pact, to resolve the 2009 Russia–Ukraine gas dispute; it extended the stay to 2042 with an option to renew.[39]

Legality and declaration of war
Further information: On conducting a special military operation
No formal declaration of war has been issued in the ongoing Russo-Ukrainian War. When Putin announced the 2022 Russian invasion of Ukraine, he claimed to commence a "special military operation", side-stepping a formal declaration of war.[40] The statement was, however, regarded as a declaration of war by the Ukrainian government[41] and reported as such by many international news sources.[42][43] While the Ukrainian parliament refers to Russia as a "terrorist state" in regard to its military actions in Ukraine,[44] it has not issued a formal declaration of war on its behalf.

The Russian invasion of Ukraine violated international law (including the Charter of the United Nations).[52][53][54][55] The invasion has also been called a crime of aggression under international criminal law[56] and under some countries' domestic criminal codes – including those of Ukraine and Russia – although procedural obstacles exist to prosecutions under these laws.[57][58]

History
Russian annexation of Crimea (2014)
For a chronological guide, see Timeline of the annexation of Crimea by the Russian Federation.

The Russian military buildup along Ukraine's eastern border in February–March 2014

The blockade of military units of the Armed Forces of Ukraine during the capture of Crimea by Russia in February–March 2014

Russian troops blocking the Ukrainian military base in Perevalne
On 20 February 2014, Russia began an annexation of Crimea.[59][60][61][62] On 22 and 23 February, under the relative power vacuum immediately after the ousting of Viktor Yanukovich,[63] Russian troops and special forces began moving into Crimea through Novorossiysk.[61] On 27 February, Russian forces without insignias began their advance into the Crimean Peninsula.[64] They took strategic positions and captured the Crimean Parliament, raising a Russian flag. Security checkpoints isolated the Crimean Peninsula from the rest of Ukraine and restricted movement within the territory.[65][66][67][68]

In the following days, Russian soldiers secured key airports and a communications center.[69] Russian cyberattacks shut down websites associated with the Ukrainian government, news media, and social media. Cyberattacks also enabled Russian access to the mobile phones of Ukrainian officials and members of parliament, further disrupting communications.[70]

On 1 March, the Russian legislature approved the use of armed forces, leading to an influx of Russian troops and military hardware into the peninsula.[69] In the following days, all remaining Ukrainian military bases and installations were surrounded and besieged, including the Southern Naval Base. After Russia formally annexed the peninsula on 18 March, Ukrainian military bases and ships were stormed by Russian forces. On 24 March, Ukraine ordered troops to withdraw; by 30 March, all Ukrainian forces had left the peninsula.

On 15 April, the Ukrainian parliament declared Crimea a territory temporarily occupied by Russia.[71] After the annexation, the Russian government increased its military presence in the region and made nuclear threats.[72] Putin said that a Russian military task force would be established in Crimea.[73] In November, NATO stated that it believed Russia was deploying nuclear-capable weapons to Crimea.[74] Since the annexation of Crimea, certain NATO members have been providing training for the Ukrainian army.[75]

War in the Donbas (2014–2015)
For a chronological guide, see Timeline of the war in Donbas (2014).
See also: Combatants of the war in Donbas and List of equipment used by Russian separatist forces of the war in Donbas

Ukrainian troops deploy in response to Russian maneuvers. Early March 2014.
Pro-Russia unrest
Main article: 2014 pro-Russian unrest in Ukraine
Beginning in late February 2014, demonstrations by pro-Russian and anti-government groups took place in major cities across the eastern and southern regions of Ukraine.[76] The first protests across southern and eastern Ukraine were largely native expressions of discontent with the new Ukrainian government.[76][77] Russian involvement at this stage was limited to voicing support for the demonstrations.[77][78] Russia exploited this, however, launching a coordinated political and military campaign against Ukraine.[77][79] Putin gave legitimacy to the separatists when he described the Donbas as part of "New Russia" (Novorossiya), and expressed bewilderment as to how the region had ever become part of Ukraine.[80]

In late March, Russia continued to gather forces near the Ukrainian eastern border, reaching 30–40,000 troops by April.[81][34] The deployment was used to threaten escalation and disrupt Ukraine's response.[34] This threat forced Ukraine to divert forces to its borders instead of the conflict zone.[34]

Ukrainian authorities cracked down on the pro-Russian protests and arrested local separatist leaders in early March. Those leaders were replaced by people with ties to the Russian security services and interests in Russian businesses.[82] By April 2014, Russian citizens had taken control of the separatist movement, supported by volunteers and materiel from Russia, including Chechen and Cossack fighters.[83][84][85][86] According to Donetsk People's Republic (DPR) commander Igor Girkin, without this support in April, the movement would have dissipated, as it had in Kharkiv and Odesa.[87] The separatist groups held disputed referendums in May[88][89][90] which were not recognised by Ukraine or any other UN member state.[88]

Armed conflict

Ukrainian response to Russian activities in Donbas after seizure of Sloviansk on 12 April. April-May 2014.
In April, armed conflict began in eastern Ukraine between Russian-backed separatist forces and Ukraine. The separatists declared the People's Republics of Donetsk and Luhansk. From 6 April, militants occupied government buildings in many cities and took control of border crossings to Russia, transport hubs, a broadcasting center, and other strategic infrastructure. On 12 April several armed groups took cities of Sloviansk, Kramatorsk and then Horlivka, Druzhkivka in subsequent days. They were lead by people like retired Russian colonel Igor Girkin, lieutenant colonel Igor Bezler. Faced with continued expansion of separatist territorial control, on 15 April the interim Ukrainian government launched an "Anti-Terrorist Operation" (ATO), however, Ukrainian forces were poorly prepared and ill-positioned and the operation quickly stalled.[91]

By the end of April, Ukraine announced it had lost control of the provinces of Donetsk and Luhansk. It claimed to be on "full combat alert" against a possible Russian invasion and reinstated conscription to its armed forces.[92] Through May, the Ukrainian campaign focused on containing the separatists by securing key positions around the ATO zone to position the military for a decisive offensive once Ukraine's national mobilization had completed.

As conflict between the separatists and the Ukrainian government escalated in May, Russia began to employ a "hybrid approach", combining disinformation tactics, irregular fighters, regular Russian troops, and conventional military support.[93][94][95] The First Battle of Donetsk Airport followed the Ukrainian presidential elections. It marked a turning point in conflict; it was the first battle between the separatists and the Ukrainian government that involved large numbers of Russian "volunteers".[96][97]: 15  According to Ukraine, at the height of the conflict in the summer of 2014, Russian paramilitaries made up between 15% and 80% of the combatants.[85] From June Russia trickled in arms, armor, and munitions.

On 17 July 2014, Russian controlled forces shot down a passenger aircraft, Malaysia Airlines Flight 17, as it was flying over eastern Ukraine.[98] Investigations and the recovery of bodies began in the conflict zone as fighting continued.[99][100][101]

By the end of July, Ukrainian forces were pushing into cities, to cut off supply routes between the two, isolating Donetsk and attempting to restore control of the Russo-Ukrainian border. By 28 July, the strategic heights of Savur-Mohyla were under Ukrainian control, along with the town of Debaltseve, an important railroad hub.[102] These operational successes of Ukrainian forces threatened the existence of the DPR and LPR statelets, prompting Russian cross-border shelling targeted against Ukrainian troops on their own soil, from mid-July onwards.[103]

August 2014 Russian invasion
See also: Battle of Ilovaisk

June–August 2014 progression map
After a series of military defeats and setbacks for the separatists, who united under the banner of "Novorossiya",[104][105] Russia dispatched what it called a "humanitarian convoy" of trucks across the border in mid-August 2014. Ukraine called the move a "direct invasion".[106] Ukraine's National Security and Defence Council reported that convoys were arriving almost daily in November (up to 9 convoys on 30 November) and that their contents were mainly arms and ammunition. Strelkov claimed that in early August, Russian servicemen, supposedly on "vacation" from the army, began to arrive in Donbas.[107]

By August 2014, the Ukrainian "Anti-Terrorist Operation" shrank the territory under pro-Russian control, and approached the border.[108] Igor Girkin urged Russian military intervention, and said that the combat inexperience of his irregular forces, along with recruitment difficulties amongst the local population, had caused the setbacks. He stated, "Losing this war on the territory that President Vladimir Putin personally named New Russia would threaten the Kremlin's power and, personally, the power of the president".[109]

In response to the deteriorating situation, Russia abandoned its hybrid approach, and began a conventional invasion on 25 August 2014.[108][110] On the following day, the Russian Defence Ministry said these soldiers had crossed the border "by accident".[111][112][113] According to Nikolai Mitrokhin's estimates, by mid-August 2014 during the Battle of Ilovaisk, between 20,000 and 25,000 troops were fighting in the Donbas on the separatist side, and only 40–45% were "locals".[114]

On 24 August 2014, Amvrosiivka was occupied by Russian paratroopers,[115] supported by 250 armoured vehicles and artillery pieces.[116] The same day, Ukrainian President Petro Poroshenko referred to the operation as Ukraine's "Patriotic War of 2014" and a war against external aggression.[117][118] On 25 August, a column of Russian military vehicles was reported to have crossed into Ukraine near Novoazovsk on the Azov sea coast. It appeared headed towards Ukrainian-held Mariupol,[119][120][121][122][123] in an area that had not seen pro-Russian presence for weeks.[124] Russian forces captured Novoazovsk.[125] and Russian soldiers began deporting Ukrainians who did not have an address registered within the town.[126] Pro-Ukrainian anti-war protests took place in Mariupol.[126][127] The UN Security Council called an emergency meeting.[128]


Residents of Kyiv with Sich Battalion volunteers on 26 August 2014
The Pskov-based 76th Guards Air Assault Division allegedly entered Ukrainian territory in August and engaged in a skirmish near Luhansk, suffering 80 dead. The Ukrainian Defence Ministry said that they had seized two of the unit's armoured vehicles near Luhansk, and reported destroying another three tanks and two armoured vehicles in other regions.[129][130] The Russian government denied the skirmish took place,[130] but on 18 August, the 76th was awarded the Order of Suvorov, one of Russia's highest awards, by Russian minister of defence Sergey Shoigu for the "successful completion of military missions" and "courage and heroism".[130]

The speaker of Russia's upper house of parliament and Russian state television channels acknowledged that Russian soldiers entered Ukraine, but referred to them as "volunteers".[131] A reporter for Novaya Gazeta, an opposition newspaper in Russia, stated that the Russian military leadership paid soldiers to resign their commissions and fight in Ukraine in the early summer of 2014, and then began ordering soldiers into Ukraine.[132] Russian opposition MP Lev Shlosberg made similar statements, although he said combatants from his country are "regular Russian troops", disguised as units of the DPR and LPR.[133]

In early September 2014, Russian state-owned television channels reported on the funerals of Russian soldiers who had died in Ukraine, but described them as "volunteers" fighting for the "Russian world". Valentina Matviyenko, a top United Russia politician, also praised "volunteers" fighting in "our fraternal nation".[131] Russian state television for the first time showed the funeral of a soldier killed fighting in Ukraine.[134]

Mariupol offensive and first Minsk ceasefire
Main articles: Offensive on Mariupol (September 2014) and Minsk agreements

A map of the line of control and buffer zone established by the Minsk Protocol on 5 September 2014
On 3 September, Poroshenko said he and Putin had reached a "permanent ceasefire" agreement.[135] Russia denied this, denying that it was a party to the conflict, adding that "they only discussed how to settle the conflict".[136][137] Poroshenko then recanted.[138][139] On 5 September Russia's Permanent OSCE Representative Andrey Kelin, said that it was natural that pro-Russian separatists "are going to liberate" Mariupol. Ukrainian forces stated that Russian intelligence groups had been spotted in the area. Kelin said 'there might be volunteers over there.'[140] On 4 September 2014, a NATO officer said that several thousand regular Russian forces operating in Ukraine.[141]

On 5 September 2014, the Minsk Protocol ceasefire agreement drew a line of demarcation between Ukraine and separatist-controlled portions of Donetsk and Luhansk Oblasts.

End of 2014 and Minsk II agreement
See also: 2014 Russian cross-border shelling of Ukraine
On 7 and 12 November, NATO officials reconfirmed the Russian presence, citing 32 tanks, 16 howitzer cannons and 30 trucks of troops entering the country.[142] US general Philip M. Breedlove said "Russian tanks, Russian artillery, Russian air defence systems and Russian combat troops" had been sighted.[74][143] NATO said it had seen an increase in Russian tanks, artillery pieces and other heavy military equipment in Ukraine and renewed its call for Moscow to withdraw its forces.[144] The Chicago Council on Global Affairs stated that Russian separatists enjoyed technical advantages over the Ukrainian army since the large inflow of advanced military systems in mid-2014: effective anti-aircraft weapons ("Buk", MANPADS) suppressed Ukrainian air strikes, Russian drones provided intelligence, and Russian secure communications system disrupted Ukrainian communications intelligence. The Russian side employed electronic warfare systems that Ukraine lacked. Similar conclusions about the technical advantage of the Russian separatists were voiced by the Conflict Studies Research Centre.[145] In the 12 November United Nations Security Council meeting, the United Kingdom's representative accused Russia of intentionally constraining OSCE observation missions' capabilities, pointing out that the observers were allowed to monitor only two kilometers of border, and drones deployed to extend their capabilities were jammed or shot down.[146][non-primary source needed]


Pro-Russian rebels in Donetsk in May 2015. Ukraine declared the Russian-backed separatist republics from eastern Ukraine to be terrorist organizations.[147]
In January 2014, Donetsk, Luhansk, and Mariupol represented the three battle fronts.[148] Poroshenko described a dangerous escalation on 21 January amid reports of more than 2,000 additional Russian troops, 200 tanks and armed personnel carriers crossing the border. He abbreviated his visit to the World Economic Forum because of his concerns.[149]

A new package of measures to end the conflict, known as Minsk II, was agreed on 15 February 2015.[150] On 18 February, Ukrainian forces withdrew from Debatlseve, in the last high-intensity battle of the Donbas war until 2022. In September 2015 the United Nations Human Rights Office estimated that 8000 casualties had resulted from the conflict.[151]

Line of conflict stabilizes (2015–2021)
Further information: Timeline of the war in Donbas (2015), Timeline of the war in Donbas (2016), and Timeline of the war in Donbas (2017)
After the Minsk agreements, the war settled into static trench warfare around the agreed line of contact, with few changes in territorial control. The conflict was marked by artillery duels, special forces operations, and trench warfare. Hostilities never ceased for a substantial period of time, but continued at a low level despite repeated attempts at ceasefire. In the months after the fall of Debaltseve, minor skirmishes continued along the line of contact, but no territorial changes occurred. Both sides began fortifying their position by building networks of trenches, bunkers and tunnels, turning the conflict into static trench warfare.[152][153] The relatively static conflict was labelled a "frozen" by some,[154] but Russia never achieved this as the fighting never stopped.[155][156] Between 2014 and 2022 there were 29 ceasefires, each agreed to remain in force indefinitely. However, none of them lasted more than two weeks.[157]

US and international officials continued to report the active presence of Russian military in eastern Ukraine, including in the Debaltseve area.[158] In 2015, Russian separatist forces were estimated to number around 36,000 troops (compared to 34,000 Ukrainian), of whom 8,500–10,000 were Russian soldiers. Additionally, around 1,000 GRU troops were operating in the area.[159] Another 2015 estimate held that Ukrainian forces outnumbered Russian forces 40,000 to 20,000.[160] In 2017, on average one Ukrainian soldier died in combat every three days,[161] with an estimated 6,000 Russian and 40,000 separatist troops in the region.[162][163]


Casualties of the war in Donbas
Cases of killed and wounded Russian soldiers were discussed in local Russian media.[164] Recruiting for Donbas was performed openly via veteran and paramilitary organisations. Vladimir Yefimov, leader of one such organisation, explained how the process worked in the Ural area. The organisation recruited mostly army veterans, but also policemen, firefighters etc. with military experience. The cost of equipping one volunteer was estimated at 350,000 rubles (around $6500) plus salary of 60,000 to 240,000 rubles per month.[165] The recruits received weapons only after arriving in the conflict zone. Often, Russian troops traveled disguised as Red Cross personnel.[166][167][168][169] Igor Trunov, head of the Russian Red Cross in Moscow, condemned these convoys, saying they complicated humanitarian aid delivery.[170] Russia refused to allow OSCE to expand its mission beyond two border crossings.[171]

The volunteers were issued a document claiming that their participation was limited to "offering humanitarian help" to avoid Russian mercenary laws. Russia's anti-mercenary legislation defined a mercenary as someone who "takes part [in fighting] with aims counter to the interests of the Russian Federation".[165]

In August 2016, the Ukrainian intelligence service, the SBU, published telephone intercepts from 2014 of Sergey Glazyev (Russian presidential adviser), Konstantin Zatulin, and other people in which they discussed covert funding of pro-Russian activists in Eastern Ukraine, the occupation of administration buildings and other actions that triggered the conflict.[172] As early as February 2014, Glazyev gave direct instructions to various pro-Russian parties on how to take over local administration offices, what to do afterwards, how to formulate demands, and promised support from Russia, including "sending our guys".[173][174][175]


Russian-backed separatists in May 2016
2018 Kerch Strait incident
Main article: Kerch Strait incident
See also: List of Black Sea incidents involving Russia and Ukraine and Timeline of the war in Donbas (2018)

The Kerch Strait incident over the passage between the Black and Azov seas
Russia gained de facto control of the Kerch Strait in 2014. In 2017, Ukraine appealed to a court of arbitration over the use of the strait. By 2018 Russia had built a bridge over the strait, limiting the size of ships that could pass through, imposed new regulations, and repeatedly detained Ukrainian vessels.[176] On 25 November 2018, three Ukrainian boats traveling from Odesa to Mariupol were seized by Russian warships; 24 Ukrainian sailors were detained.[177][178] A day later on 26 November 2018, the Ukrainian parliament overwhelmingly backed the imposition of martial law along Ukraine's coastal regions and those bordering Russia.[179]

2019–2020
Further information: Timeline of the war in Donbas (2019) and Timeline of the war in Donbas (2020)

From left, Russian President Vladimir Putin, French President Emmanuel Macron, German Chancellor Angela Merkel and Ukrainian President Volodymyr Zelenskyy in Paris, France, December 2019
More than 110 Ukrainian soldiers were killed in the conflict in 2019.[180] In May 2019, newly elected Ukrainian President Volodymyr Zelenskyy took office promising to end the war in Donbas.[180] In December 2019, Ukraine and pro-Russian separatists began swapping prisoners of war. Around 200 prisoners were exchanged on 29 December 2019.[181][182][183][184] According to Ukrainian authorities, 50 Ukrainian soldiers were killed in 2020.[185] Since 2019, Russia has issued over 650,000 internal Russian passports to Ukrainians.[186][187]

Russian military buildup around Ukraine (2021–2022)
Main article: Prelude to the 2022 Russian invasion of Ukraine
Further information: Timeline of the war in Donbas (2021) and Timeline of the war in Donbas (2022)
From March to April 2021, Russia commenced a major military build-up near the border, followed by a second build-up between October 2021 to February 2022 in Russia and Belarus.[188] Throughout, the Russian government repeatedly denied it had plans to attack Ukraine.[189][190]

In early December 2021, following Russian denials, the US released intelligence of Russian invasion plans, including satellite photographs showing Russian troops and equipment near the border.[191] The intelligence reported a Russian list of key sites and individuals to be killed or neutralized.[192] The US released multiple reports that accurately predicted the invasion plans.[192]

Russian accusations and demands
Further information: Disinformation in the 2022 Russian invasion of Ukraine and Russian irredentism

Ukrainian deputy prime minister Olha Stefanishyna with NATO secretary-general Jens Stoltenberg at a conference on 10 January 2022 regarding a potential Russian invasion
In the months preceding the invasion, Russian officials accused Ukraine of inciting tensions, Russophobia, and repressing Russian speakers. They made multiple security demands of Ukraine, NATO, and other EU countries. On 9 December 2021 Putin said that "Russophobia is a first step towards genocide".[193][194] Putin's claims were dismissed by the international community,[195] and Russian claims of genocide were rejected as baseless.[196][197][198] In a 21 February speech,[199] Putin questioned the legitimacy of the Ukrainian state, repeating an inaccurate claim that "Ukraine never had a tradition of genuine statehood".[200] He incorrectly stated that Vladimir Lenin had created Ukraine, by carving a separate Soviet Republic out of what Putin said was Russian land, that Joseph Stalin extended Ukrainian territory with lands from other eastern European countries following the Second World War, and that Nikita Khrushchev "took Crimea away from Russia for some reason and gave it to Ukraine" in 1954.[201]

Putin falsely claimed that Ukrainian society and government were dominated by neo-Nazism, invoking the history of collaboration in German-occupied Ukraine during World War II,[202][203] and echoing an antisemitic conspiracy theory that cast Russian Christians, rather than Jews, as the true victims of Nazi Germany.[204][195] Ukraine does suffer a far-right fringe, including the neo-Nazi linked Azov Battalion and Right Sector.[205][203] Analysts described Putin's rhetoric as greatly exaggerated.[206][202] Zelenskyy, who is Jewish, stated that his grandfather served in the Soviet army fighting against the Nazis;[207] three of his family members were killed in the Holocaust.[206]


A U.S. intelligence assessment map and imagery on Russian military movement nearby the Ukrainian border, as on 3 December 2021. It assessed that Russia had deployed about 70,000 military personnel mostly about 100–200 kilometres (62–124 mi) from the Ukrainian border, with an assessment this could be increased to 175,000 personnel. Published by The Washington Post.[208]
During the second build-up, Russia issued demands to the US and NATO, insisting on a legally-binding agreement preventing Ukraine from ever joining NATO, and the removal of multinational forces stationed in NATO's Eastern European member states.[209] These demands were rejected.[210] A treaty to prevent Ukraine joining NATO would go against the alliance's "open door" policy, although NATO made no efforts to comply with Ukraine's requests to join.[211] NATO Secretary General Jens Stoltenberg replied that "Russia has no say" on whether Ukraine joins, and that "Russia has no right to establish a sphere of influence to try to control their neighbors".[212]

Prelude to full invasion
Fighting in Donbas escalated significantly from 17 February 2022 onwards.[213] The Ukrainians and the pro-Russian separatists each accused the other of attacks.[214][215] There was a sharp increase in artillery shelling by the Russian-led militants in Donbas, which was considered by Ukraine and its supporters to be an attempt to provoke the Ukrainian army or create a pretext for invasion.[216][217][218] On 18 February, the Donetsk and Luhansk people's republics ordered mandatory emergency evacuations of civilians from their respective capital cities,[219][220][221] although observers noted that full evacuations would take months.[222] The Russian government intensified its disinformation campaign, with Russian state media promoting fabricated videos (false flags) on a nearly hourly basis purporting to show Ukrainian forces attacking Russia.[223] Many of the disinformation videos were amateurish, and evidence showed that the claimed attacks, explosions, and evacuations in Donbas were staged by Russia.[223][224][225]

Putin's address to the nation on 21 February (English subtitles available)
On 21 February at 22:35 (UTC+3),[226] Putin announced that the Russian government would diplomatically recognize the Donetsk and Luhansk people's republics.[227] The same evening, Putin directed that Russian troops deploy into Donbas, in what Russia referred to as a "peacekeeping mission".[228][229] On 22 February, the Federation Council unanimously authorised Putin to use military force outside Russia.[230] In response, Zelenskyy ordered the conscription of army reservists;[231] The following day, Ukraine's parliament proclaimed a 30-day nationwide state of emergency and ordered the mobilisation of all reservists.[232][233][234] Russia began to evacuate its embassy in Kyiv.[235]

On the night of 23 February,[236] Zelenskyy gave a speech in Russian in which he appealed to the citizens of Russia to prevent war.[237][238] He rejected Russia's claims about neo-Nazis and stated that he had no intention of attacking the Donbas.[239] Kremlin spokesman Dmitry Peskov said on 23 February that the separatist leaders in Donetsk and Luhansk had sent a letter to Putin stating that Ukrainian shelling had caused civilian deaths and appealing for military support.[240]

Full-scale Russian invasion of Ukraine (2022–present)
Main article: Russian invasion of Ukraine
For a chronological guide, see Timeline of the 2022 Russian invasion of Ukraine.

Animated map of Russia's invasion of Ukraine through 5 December 2022 (click to play animation)
The Russian invasion of Ukraine began on the morning of 24 February,[241] when Putin announced a "special military operation" to "demilitarise and denazify" Ukraine.[242][243] Minutes later, missiles and airstrikes hit across Ukraine, including Kyiv, shortly followed by a large ground invasion along multiple fronts.[244][245] Zelenskyy declared martial law and a general mobilisation of all male Ukrainian citizens between 18 and 60, who were banned from leaving the country.[246][247]

Russian attacks were initially launched on a northern front from Belarus towards Kyiv, a north-eastern front towards Kharkiv, a southern front from Crimea, and a south-eastern front from Luhansk and Donetsk.[248][249] In the northern front, amidst heavy losses and strong Ukrainian resistance surrounding Kyiv, Russia's advance stalled in March, and by April its troops retreated. On 8 April, Russia placed its forces in southern and eastern Ukraine under the command of General Aleksandr Dvornikov, and some units withdrawn from the north were redeployed to the Donbas.[250] On 19 April, Russia launched a renewed attack across a 500 kilometres (300 mi) long front extending from Kharkiv to Donetsk and Luhansk.[251] By 13 May, a Ukraine counter-offensive had driven back Russian forces near Kharkiv. By 20 May, Mariupol fell to Russian troops following a prolonged siege of the Azovstal steel works.[252][253] Russian forces continued to bomb both military and civilian targets far from the frontline.[254][255] The war caused the largest refugee and humanitarian crisis within Europe since the Yugoslav Wars in the 1990s;[256][257] the UN described it as the fastest-growing such crisis since World War II.[258] In the first week of the invasion, the UN reported over a million refugees had fled Ukraine; this subsequently rose to over 7,405,590 by 24 September, a reduction from over eight million due to some refugees' return.[259][260]


Ukrainian soldiers killed in the Russo-Ukrainian War in 2022
Ukrainian forces launched counteroffensives in the south in August, and in the northeast in September. On 30 September, Russia annexed four oblasts of Ukraine which it had partially conquered during the invasion.[261] This annexation was generally unrecognized and condemned by the countries of the world.[262] After Putin announced that he would begin conscription drawn from the 300,000 citizens with military training and potentially the pool of about 25 million Russians who could be eligible for conscription, one-way tickets out of the country nearly or completely sold out.[263][264] The Ukrainian offensive in the northeast successfully recaptured the majority of Kharkiv Oblast in September. In the course of the southern counteroffensive, Ukraine retook the city of Kherson in November and Russian forces withdrew to the east bank of the Dnieper River.[citation needed]

The invasion was internationally condemned as a war of aggression.[265][266] A United Nations General Assembly resolution demanded a full withdrawal of Russian forces, the International Court of Justice ordered Russia to suspend military operations and the Council of Europe expelled Russia. Many countries imposed new sanctions, which affected the economies of Russia and the world,[267] and provided humanitarian and military aid to Ukraine.[268] In September 2022, Putin signed a law that would punish anyone who resists conscription with a 10-year prison sentence[269] resulting in an international push to allow asylum for Russians fleeing conscription.[270]

According to an estimate published by The New York Times, as of February 2023, the "number of Russian troops killed and wounded in Ukraine is approaching 200,000."[271]

Human rights violations
See also: Casualties of the Russo-Ukrainian War, Humanitarian situation during the war in Donbas, and Russian war crimes § Ukraine
Violations of human rights and atrocity crimes have both occurred during the war. From 2014 to 2021, there were more than 3,000 civilian casualties, with most occurring in 2014 and 2015.[272] The right of movement was impeded for the inhabitants of the conflict zone.[273] Arbitrary detention was practiced by both sides in the first years of the conflict. It decreased after 2016 in government-held areas, while in the separatist-held ones it continued.[274] Investigations into the abuses committed by both sides made little progress.[275][276]

Since the beginning of the Russian invasion of Ukraine in 2022, Russian authorities and armed forces have committed multiple war crimes in the form of deliberate attacks against civilian targets,[277][278] massacres of civilians, torture and rape of women and children,[279][280] and indiscriminate attacks in densely populated areas. After the Russian withdrawal from areas north of Kyiv, overwhelming evidence of war crimes by Russian forces was discovered. In particular, in the town of Bucha, evidence emerged of a massacre of civilians perpetrated by Russian troops, including torture, mutilation, rape, looting and deliberate killings of civilians.[281][282][283] the UN Human Rights Monitoring Mission in Ukraine (OHCHR) has documented the murder of at least 73 civilians – mostly men, but also women and children – in Bucha.[284] More than 1,200 bodies of civilians were found in the Kyiv region after Russian forces withdrew, some of them summarily executed. There were reports of forced deportations of thousands of civilians, including children, to Russia, mainly from Russian-occupied Mariupol,[285][286] as well as sexual violence, including cases of rape, sexual assault and gang rape,[287] and deliberate killing of Ukrainian civilians by Russian forces.[288]

Ukrainian forces have also been accused of committing various war crimes, including mistreatment of detainees, though on a much smaller scale than Russian forces.[289][290]

Related issues
Gas disputes
See also: Russia–Ukraine gas disputes, Nord Stream, Nord Stream 2, and Russia in the European energy sector

Major Russian natural gas pipelines to Europe

  Europe TTF natural gas
Until 2014 Ukraine was the main transit route for Russian natural gas sold to Europe, which earned Ukraine about US$3 billion a year in transit fees, making it the country's most lucrative export service.[291] Following Russia's launch of the Nord Stream pipeline, which bypasses Ukraine, gas transit volumes steadily decreased.[291] Following the start of the Russo-Ukrainian War in February 2014, severe tensions extended to the gas sector.[292][293] The subsequent outbreak of war in the Donbas region forced the suspension of a project to develop Ukraine's own shale gas reserves at the Yuzivska gas field, which had been planned as a way to reduce Ukrainian dependence on Russian gas imports.[294] Eventually, the EU commissioner for energy Günther Oettinger was called in to broker a deal securing supplies to Ukraine and transit to the EU.[295]

An explosion damaged a Ukrainian portion of the Urengoy–Pomary–Uzhhorod pipeline in Ivano-Frankivsk Oblast in May 2014. Ukrainian officials blamed Russian terrorists.[296] Another section of the pipeline exploded in the Poltava Oblast on 17 June 2014, one day after Russia limited the supply of gas to Ukrainian customers due to non-payment. Ukraine's Interior Minister Arsen Avakov said the following day that the explosion had been caused by a bomb.[297]

In 2015, Russian state media reported that Russia planned to completely abandon gas supplies to Europe through Ukraine after 2018.[298][299] Russia's state-owned energy giant Gazprom had already substantially reduced the volumes of gas transited across Ukraine, and expressed its intention to reduce the level further by means of transit-diversification pipelines (Turkish Stream, Nord Stream, etc.).[300] Gazprom and Ukraine agreed to a five-year deal on Russian gas transit to Europe at the end of 2019.[301][302]

In 2020, the TurkStream natural gas pipeline running from Russia to Turkey changed the regional gas flows in South-East Europe by diverting the transit through Ukraine and the Trans Balkan Pipeline system.[303][304]

In May 2021, the Biden administration waived Trump's CAATSA sanctions on the company behind Russia's Nord Stream 2 gas pipeline to Germany.[305][306] Ukrainian President Zelenskyy said he was "surprised" and "disappointed" by Joe Biden's decision.[307] In July 2021, the U.S. urged Ukraine not to criticise a forthcoming agreement with Germany over the pipeline.[308][309]

In July 2021, Biden and German Chancellor Angela Merkel concluded a deal that the U.S. might trigger sanctions if Russia used Nord Stream as a "political weapon". The deal aimed to prevent Poland and Ukraine from being cut off from Russian gas supplies. Ukraine will get a $50 million loan for green technology until 2024 and Germany will set up a billion dollar fund to promote Ukraine's transition to green energy to compensate for the loss of the gas-transit fees. The contract for transiting Russian gas through Ukraine will be prolonged until 2034, if the Russian government agrees.[310][311][312]

In August 2021, Zelenskyy warned that the Nord Stream 2 natural gas pipeline between Russia and Germany was "a dangerous weapon, not only for Ukraine but for the whole of Europe."[313][314] In September 2021, Ukraine's Naftogaz CEO Yuriy Vitrenko accused Russia of using natural gas as a "geopolitical weapon".[315] Vitrenko stated that "A joint statement from the United States and Germany said that if the Kremlin used gas as a weapon, there would be an appropriate response. We are now waiting for the imposition of sanctions on a 100% subsidiary of Gazprom, the operator of Nord Stream 2."[316]

Hybrid warfare
The Russo-Ukrainian conflict has also included elements of hybrid warfare using non-traditional means. Cyberwarfare has been used by Russia in operations including successful attacks on the Ukrainian power grid in December 2015 and in December 2016, which was the first successful cyber attack on a power grid,[317] and the Mass hacker supply-chain attack in June 2017, which the US claimed was the largest known cyber attack.[318] In retaliation, Ukrainian operations have included the Surkov Leaks in October 2016 which released 2,337 e-mails in relation to Russian plans for seizing Crimea from Ukraine and fomenting separatist unrest in Donbas.[319] The Russian information war against Ukraine has been another front of hybrid warfare waged by Russia.

A Russian fifth column in Ukraine has also been claimed to exist among the Party of Regions, the Communist Party, the Progressive Socialist Party and the Russian Orthodox Church.[320][321][322]

Russian propaganda and disinformation campaigns
Main articles: Russian information war against Ukraine and Disinformation in the Russian invasion of Ukraine

Pro-Kremlin TV and radio host Vladimir Solovyov voiced support for his country's invasion of Ukraine.[323]
False stories have been used to provoke public outrage during the war. In April 2014, Russian news channels Russia-1 and NTV showed a man saying he was attacked by a fascist Ukrainian gang on one channel and on the other channel saying he was funding the training of right-wing anti-Russia radicals.[324][325] A third segment portrayed the man as a neo-Nazi surgeon.[326] In May 2014, Russia-1 aired a story about Ukrainian atrocities using footage of a 2012 Russian operation in North Caucasus.[327] In the same month, the Russian news network Life presented a 2013 photograph of a wounded child in Syria as a victim of Ukrainian troops who had just retaken Donetsk International Airport.[328]

In June 2014, several Russian state news outlets reported that Ukraine was using white phosphorus using 2004 footage of white phosphorus being used by the United States in Iraq.[327] In July 2014, Channel One Russia broadcast an interview with a woman who said that a 3-year-old boy who spoke Russian was crucified by Ukrainian nationalists in a fictitious square in Sloviansk that turned out to be false.[329][330][325][327]

In 2022, Russian state media told stories of genocide and mass graves full of ethnic Russians in eastern Ukraine. One set of graves outside Luhansk was dug when intense fighting in 2014 cut off the electricity in the local morgue. Amnesty International investigated 2014 Russian claims of mass graves filled with hundreds of bodies and instead found isolated incidents of extrajudicial executions by both sides.[331][332][333]


Russian artist Alexandra Skochilenko was arrested for replacing price tags in supermarkets with anti-war messages.[334]
The Russian censorship apparatus Roskomnadzor ordered the country's media to employ information only from Russian state sources or face fines and blocks,[335] and ordered media and schools to describe the war as a "special military operation".[336] On 4 March 2022, Putin signed into law a bill introducing prison sentences of up to 15 years for those who publish "fake news" about the Russian military and its operations,[337] leading to some media outlets to stop reporting on Ukraine.[338] Russia's opposition politician Alexei Navalny said the "monstrosity of lies" in the Russian state media "is unimaginable. And, unfortunately, so is its persuasiveness for those who have no access to alternative information."[339] He tweeted that "warmongers" among Russian state media personalities "should be treated as war criminals. From the editors-in-chief to the talk show hosts to the news editors, [they] should be sanctioned now and tried someday."[340]

Putin and Russian media have described the government of Ukraine as being led by neo-Nazis persecuting ethnic Russians who are in need of protection by Russia, despite Ukraine's President Zelenskyy being Jewish.[341][342][332] According to journalist Natalia Antonova, "Russia's present-day war of aggression is refashioned by propaganda into a direct continuation of the legacy of the millions of Russian soldiers who died to stop" Nazi Germany in World War II.[343] Ukraine's rejection of the adoption of Russia-initiated General Assembly resolutions on combating the glorification of Nazism, the latest iteration of which is General Assembly Resolution A/C.3/76/L.57/Rev.1 on Combating Glorification of Nazism, Neo-Nazism and other Practices that Contribute to Fueling Contemporary Forms of Racism, Racial Discrimination, Xenophobia and Related Intolerance, serve to present Ukraine as a pro-Nazi state, and indeed likely forms the basis for Russia's claims, with the only other state rejecting the adoption of the resolution being the US.[344][345] The Deputy US Representative for ECOSOC describes such resolutions as "thinly veiled attempts to legitimize Russian disinformation campaigns denigrating neighboring nations and promoting the distorted Soviet narrative of much of contemporary European history, using the cynical guise of halting Nazi glorification".[346]

NAFO ('North Atlantic Fellas Organization'), a loose cadre of online 'shitposters' vowing to fight Russian disinformation generally identified by cartoon Shiba Inu dogs in social media, gained notoriety after June 2022, in the wake of a Twitter quarrel with Russian diplomat Mikhail Ulyanov.[347]

Russia–NATO relations
Main article: Russia–NATO relations
In his speech justifying the invasion of Ukraine, Putin falsely claimed that NATO military infrastructure was being built up inside Ukraine and was a threat to Russia.[348] Russian Foreign Minister Sergey Lavrov characterized the conflict as a proxy war started by NATO.[349] He said: "We don't think we're at war with NATO ... Unfortunately, NATO believes it is at war with Russia".[350] NATO says it is not at war with Russia; its official policy is that it does not seek confrontation, but rather its members support Ukraine in "its right to self-defense, as enshrined in the UN Charter".[351] NATO and Russia had co-operated until Russia annexed Crimea.[351] Former CIA director Leon Panetta told the ABC that the U.S. is 'without question' involved in a proxy war with Russia.[352]

Russian military aircraft flying over the Baltic and Black Seas often do not indicate their position or communicate with air traffic controllers, thus posing a potential risk to civilian airliners. NATO aircraft scrambled many times to track and intercept these aircraft near alliance airspace. The Russian aircraft intercepted never entered NATO airspace, and the interceptions were conducted in a safe and routine manner.[353]

International reactions
Further information: International sanctions during the Russo-Ukrainian War and List of military aid to Ukraine during the Russo-Ukrainian War
See also: Second Cold War
Reactions to the Russian annexation of Crimea
Main article: International reactions to the annexation of Crimea by the Russian Federation
Ukrainian response

Following Russia's annexation of Crimea, Ukraine blocked the North Crimean Canal, which provided 85% of Crimea's drinking and irrigation water.[354]
Interim Ukrainian President Oleksandr Turchynov accused Russia of "provoking a conflict" by backing the seizure of the Crimean parliament building and other government offices on the Crimean peninsula. He compared Russia's military actions to the 2008 Russo-Georgian War, when Russian troops occupied parts of the Republic of Georgia and the breakaway enclaves of Abkhazia and South Ossetia were established under the control of Russian-backed administrations. He called on Putin to withdraw Russian troops from Crimea and stated that Ukraine will "preserve its territory" and "defend its independence".[355] On 1 March, he warned, "Military intervention would be the beginning of war and the end of any relations between Ukraine and Russia."[356] On 1 March, Acting President Oleksandr Turchynov placed the Armed Forces of Ukraine on full alert and combat readiness.[357]

The Ministry of Temporarily Occupied Territories and IDPs was established by Ukrainian government on 20 April 2016 to manage occupied parts of Donetsk, Luhansk and Crimea regions affected by Russian military intervention of 2014.[358]

NATO and United States military response
Further information: Operation Atlantic Resolve, European Deterrence Initiative, NATO Enhanced Forward Presence, and Russia–NATO relations

A U.S. Army convoy in Vilseck, Germany during Operation Atlantic Resolve, NATO's efforts to reassert its military presence in central and eastern Europe that began in April 2014.
On 4 March 2014, the United States pledged $1 billion in aid to Ukraine.[359] Russia's actions increased tensions in nearby countries historically within its sphere of influence, particularly the Baltic and Moldova. All have large Russian-speaking populations, and Russian troops are stationed in the breakaway Moldovan territory of Transnistria.[360] Some devoted resources to increasing defensive capabilities,[361] and many requested increased support from the U.S. and the North Atlantic Treaty Organization, which they had joined in recent years.[360][361] The conflict "reinvigorated" NATO, which had been created to face the Soviet Union, but had devoted more resources to "expeditionary missions" in recent years.[362]

In addition to diplomatic support in its conflict with Russia, the U.S. provided Ukraine with US$1.5 billion in military aid during the 2010s.[363] In 2018 the U.S. House of Representatives passed a provision blocking any training of Azov Battalion of the Ukrainian National Guard by American forces. In previous years, between 2014 and 2017, the U.S. House of Representatives passed amendments banning support of Azov, but due to pressure from the Pentagon, the amendments were quietly lifted.[364][365][366]

Financial markets

Euro/RUB exchange rate

USD/Russian Ruble Exchange Rate
The initial reaction to the escalation of tensions in Crimea caused the Russian and European stock market to tumble.[367] The intervention caused the Swiss franc to climb to a 2-year high against the dollar and 1-year high against the Euro. The Euro and the US dollar both rose, as did the Australian dollar.[368] The Russian stock market declined by more than 10 percent, while the Russian ruble hit all-time lows against the US dollar and the Euro.[369][370][371] The Russian central bank hiked interest rates and intervened in the foreign exchange markets to the tune of $12 billion[clarification needed] to try to stabilize its currency.[368] Prices for wheat and grain rose, with Ukraine being a major exporter of both crops.[372]

Later in March 2014, the reaction of the financial markets to the Crimea annexation was surprisingly mellow, with global financial markets rising immediately after the referendum held in Crimea, one explanation being that the sanctions were already priced in following the earlier Russian incursion.[373] Other observers considered that the positive reaction of the global financial markets on Monday 17 March 2014, after the announcement of sanctions against Russia by the EU and the US, revealed that these sanctions were too weak to hurt Russia.[374] In early August 2014, the German DAX was down by 6 percent for the year, and 11 percent since June, over concerns Russia, Germany's 13th biggest trade partner, would retaliate against sanctions.[375]

Reactions to the Russian intervention in the Donbas
Further information: International reactions to the war in Donbas

Peace march in Moscow, 21 September 2014

Pro-Russian supporters in Donetsk, 20 December 2014
Ukrainian public opinion
See also: Putin khuylo!
A poll of the Ukrainian public, excluding Russian-annexed Crimea, was taken by the International Republican Institute from 12 to 25 September 2014.[376] 89% of those polled opposed 2014 Russian military intervention in Ukraine. As broken down by region, 78% of those polled from Eastern Ukraine (including Dnipropetrovsk Oblast) opposed said intervention, along with 89% in Southern Ukraine, 93% in Central Ukraine, and 99% in Western Ukraine.[376] As broken down by native language, 79% of Russian speakers and 95% of Ukrainian speakers opposed the intervention. 80% of those polled said the country should remain a unitary country.[376]

A poll of the Crimean public in Russian-annexed Crimea was taken by the Ukrainian branch of Germany's biggest market research organization, GfK, on 16–22 January 2015. According to its results: "Eighty-two percent of those polled said they fully supported Crimea's inclusion in Russia, and another 11 percent expressed partial support. Only 4 percent spoke out against it."[377][378][379]

A joint poll conducted by Levada and the Kyiv International Institute of Sociology from September to October 2020 found that in the breakaway regions controlled by the DPR/LPR, just over half of the respondents wanted to join Russia (either with or without some autonomous status) while less than one-tenth wanted independence and 12% wanted reintegration into Ukraine. It contrasted with respondents in Kyiv-controlled Donbas, where a vast majority felt the separatist regions should be returned to Ukraine.[380] According to results from Levada in January 2022, roughly 70% of those in the breakaway regions said their territories should become part of the Russian Federation.[381]

Russian public opinion
See also: 2014 anti-war protests in Russia
An August 2014 survey by the Levada Centre reported that only 13% of those Russians polled would support the Russian government in an open war with Ukraine.[382] Street protests against the war in Ukraine arose in Russia. Notable protests first occurred in March[383][384] and large protests occurred in September when "tens of thousands" protested the war in Ukraine with a peace march in downtown Moscow on Sunday, 21 September 2014, "under heavy police supervision".[385]

Reactions to the 2022 Russian invasion of Ukraine
Main article: Reactions to the 2022 Russian invasion of Ukraine
Ukrainian public opinion

Ukrainian refugees in Kraków protest against the war, 6 March 2022
In March 2022, a week after the Russian invasion of Ukraine, 98% of Ukrainians – including 82% of ethnic Russians living in Ukraine – said they did not believe that any part of Ukraine was rightfully part of Russia, according to Lord Ashcroft's polls which did not include Crimea and the separatist-controlled part of Donbas. 97% of Ukrainians said they had an unfavourable view of Russian President Vladimir Putin, with a further 94% saying they had an unfavourable view of the Russian Armed Forces.[386]

At the end of 2021, 75% of Ukrainians said they had a positive attitude toward ordinary Russians, while in May 2022, 82% of Ukrainians said they had a negative attitude toward ordinary Russians.[387]

Russian public opinion
An April 2022 survey by the Levada Centre reported that approximately 74% of the Russians polled supported the "special military operation" in Ukraine, suggesting that Russian public opinion has shifted considerably since 2014.[388] According to some sources, a reason many Russians supported the "special military operation" has to do with the propaganda and disinformation.[389][390] In addition, it has been suggested that some respondents did not want to answer pollsters' questions for fear of negative consequences.[391][392] At the end of March, a poll conducted in Russia by the Levada Center concluded the following: When asked why they think the military operation is taking place, respondents said it was to protect and defend civilians, ethnic Russians or Russian speakers in Ukraine (43%), to prevent an attack on Russia (25%), to get rid of nationalists and "denazify" Ukraine (21%), and to incorporate Ukraine or the Donbas region into Russia (3%)."[393] According to polls, the Russian President's rating rose from 71% on the eve of the invasion to 82% in March 2023.[394]

United States

   Russia
   Countries on Russia's "Unfriendly Countries List". The list includes countries that have imposed sanctions against Russia for its invasion of Ukraine.[395]
On 28 April 2022, US President Joe Biden asked Congress for an additional $33 billion to assist Ukraine, including $20 billion to provide weapons to Ukraine.[396] On 5 May, Ukraine's Prime Minister Denys Shmyhal announced that Ukraine had received more than $12 billion worth of weapons and financial aid from Western countries since the start of Russia's invasion on 24 February.[397] On 21 May 2022, the United States passed legislation providing $40 billion in new military and humanitarian foreign aid to Ukraine, marking a historically large commitment of funds.[398][399] In August 2022, U.S. defense spending to counter the Russian war effort exceeded the first 5 years of war costs in Afghanistan. The Washington Post reported that new U.S. weapons delivered to the Ukrainian war front suggest a closer combat scenario with more casualties.[400] The United States looks to build "enduring strength in Ukraine" with increased arms shipments and a record-breaking $3 billion military aid package.[400]

Russian military suppliers
After expending large amounts of heavy weapons and munitions over months, the Russian Federation received combat drones and loitering munitions from Iran, deliveries of tanks and other armoured vehicles from Belarus, and reportedly planned to trade for artillery ammunition from North Korea and ballistic missiles from Iran.[401][402][403][404]

The U.S. has accused China of providing Russia with technology it needs for high-tech weapons, allegations which China has denied. The U.S. sanctioned a Chinese firm for providing satellite imagery to Russian mercenary forces fighting in Ukraine.[405]

In March 2023, Western nations had pressed the United Arab Emirates to halt re-exports of goods to Russia which had military uses, amidst allegations that the Gulf country exported 158 drones to Russia in 2022.[406] In May 2023, the U.S. accused South Africa of supplying arms to Russia in a covert naval operation,[407] allegations which have been denied by South African president Cyril Ramaphosa.[408]'''


# In[3]:


data


# In[4]:


import re


# In[5]:


# Remove newline characters
data = re.sub(r'\n', '', data)


# In[6]:


data=re.sub(r'\d+','',data)


# In[7]:


data=re.sub(r'\[]','',data)


# In[8]:


data


# In[9]:


# Remove parentheses using str.replace()
data = data.replace("(", "").replace(")", "")
data = re.sub(r'\[(f|d|c|a|e|-|'')\]', '', data)

print(data)


# In[10]:


data=data.split('.')


# In[11]:


tokenizer=tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(data)
seq=tokenizer.texts_to_sequences(data)


# In[12]:


seq


# In[13]:


tokenizer.word_index
# this will return which word indexed to which name


# In[14]:


# tokenizer.word_index[2]


# In[15]:


x=[]
y=[]
total_words_dropped=0
# x will contain input data y will contain output
for i in seq:
    if len(i)>1:
        for index in range(1,len(i)):
            x.append(i[:index])
            y.append(i[index])
    else:
        total_words_dropped += 1
        
print('total single words dropped',total_words_dropped)


# In[16]:


x[:10]


# In[17]:


y[:10]


# In[18]:


x=tf.keras.preprocessing.sequence.pad_sequences(x)
# this will add 0 (like one hot encd)


# In[19]:


x


# In[20]:


x.shape


# In[21]:


y=tf.keras.utils.to_categorical(y)


# In[22]:


y


# In[23]:


y.shape


# In[24]:


vocab_size=len(tokenizer.word_index)+1
vocab_size


# In[25]:


model=tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,14),
#     tf.keras.layers.LSTM(100),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(vocab_size,activation='softmax'),
    
])


# In[26]:


model.summary


# In[28]:


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',metrics=['accuracy'])


# In[29]:


model.fit(x,y,epochs=100)


# In[32]:


model.save('next_word_prediction')


# In[36]:


vocab_array=np.array(list(tokenizer.word_index.keys()))


# In[37]:


vocab_array


# In[38]:


def make_prediction(text,n_words):
    for i in range(n_words):
        text_tokenize=tokenizer.texts_to_sequences([text])
        text_padded=tf.keras.preprocessing.sequence.pad_sequences(text_tokenize,maxlen=14)
        prediction=np.squeeze(np.argmax(model.predict(text_padded),axis=-1))
        prediction=str(vocab_array[prediction-1])
        text += " "+ prediction
    return text


# In[47]:


make_prediction('death  ',20)


# In[ ]:




