<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
  <h1 align="center">Portuguese Named Entity Recognition for Biomedical Information Extraction
 </h1>
<br />
<br />

<!-- ABOUT THE PROJECT -->
This project provides models for the Named Entity Recognition task and algorithms to train, test, and evaluate the
proposed models.

The data are in the Portuguese language and from the biomedical domain. 
We provided only one file as a data example, but we will release the full dataset soon.

For the full documentation, please follow this link: [Docs](https://pavalucas.github.io/portuguese-ner-biomedical/). 

<br />

<!-- GETTING STARTED -->
## Installation

First, check that you have Python 3.7+:
```
python3 --version
```

 Then, you can install this package with the following instructions:

```sh
git clone https://github.com/pavalucas/portuguese-ner-biomedical.git
cd portuguese-ner-biomedical
python3 setup.py install
```
<br />

### Tests
We developed tests using [PyTest](https://pytest.org/). All test files are located under the `tests/` folder.

To run all tests:
```
pytest .
```

To output the test log to a file:
```
pytest . > testlog.txt
```
---

<br />

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


<br />

<!-- CONTACT -->
## Contact

Lucas Pavanelli - lucasapava@gmail.com

Project Link: [portuguese-ner-biomedical](https://github.com/pavalucas/portuguese-ner-biomedical)


<br />



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/github/license/dbeyda/fast-online-packing.svg?style=for-the-badge
[license-url]: https://github.com/pavalucas/portuguese-ner-biomedical/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/lucas-pavanelli-a49131a5/?locale=en_US