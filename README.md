<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/catiaspsilva/README-template">
    <img src="images/siamese-cat-isolated-on-transparent-background-ai-generated-png.png" alt="Logo" width="150" height="150">
  </a>

  <h3 align="center">Object tracking with Siamese networks</h3>




<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of contents</summary>
  <ol>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#model-architecture">Model architecture</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#challenges-and-future-work">Challenges and Future Work</a></li>
    <li><a href="#references">References</a></li>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- Introduction -->
## <span id="introduction"> Introduction </span>

This project was developed as part of the Practical Seminar in Machine Learning (PSI:ML) 10. It implements single object tracking using Siamese networks, focusing on the challenge of accurately tracking a specified target throughout a video. The target is identified in the first frame and must be consistently detected and followed in all subsequent frames.

<!-- Project Structure -->
## <span id="project-structure"> Project Structure </span>
Project structure is as follows:
<ul>
  <li>data - LaSOT images (in our case only bicycle and airplane classes of images)</li>
  <li>training - with all .py files that are already in it</li>
</ul>
Download LaSOT dataset from <a href="https://onedrive.live.com/?authkey=%21AKDCa351cL3g44Q&id=83EEFE32EECC7F4B%2133234&cid=83EEFE32EECC7F4B">LaSOT</a>. In the code training and test set are seperated via different classes ImageLASOT_train, ImageLASOT_val, ImageLASOT_test. The main difference between these classes is in the value of the attribute <i>subclasses_indexes</i>. Each class of images has 20 subclasses, and you can play around with this to determine how much parameters are going to be for training, validation and test sets.

<!-- Model arch -->
## <span id="model-architecture"> Model architecture </span>

In this section you should provide instructions on how to use this repository to recreate your project locally.

<!-- Project Structure -->
## <span id="results"> Results </span>

In this section you should provide instructions on how to use this repository to recreate your project locally.

<!-- Project Structure -->
## <span id="challenges-and-future-work"> Challenges and future work </span>

In this section you should provide instructions on how to use this repository to recreate your project locally.

<!-- Project Structure -->
## <span id="references"> References </span>

In this section you should provide instructions on how to use this repository to recreate your project locally.

<!-- Authors -->
## Authors

<ul>
  <li>
    <a href="https://github.com/0elena0">Elena Nešović </a>
  </li>
  <li>
    <a href="https://github.com/Dovlane">Vladimir Ignjatijević</a>
  </li>
</ul>


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

You can acknowledge any individual, group, institution or service.
* [Catia Silva](https://faculty.eng.ufl.edu/catia-silva/)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)


