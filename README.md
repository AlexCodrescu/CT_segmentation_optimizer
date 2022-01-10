Pentru rezolvarea problemei am inceput prin a "curata" imaginea rezultata din scanarea aparatului CT cu ajutorul valorii medie a unitatilor hounsfield din interiorul matricii rezultate 
din segmentarea manuala de catre doctor (am inlaturat valorile hounsfield care nu sunt apropiate de acea valoare),am folosit filters.median ca sa fac denoise dupa care am creat marginile
pe matricea rezultata.Am scos in "relief" marginile si am gasit punctele cele mai indepartate de margini dupa care am aplicat watershed.Pentru a lucra pe seturi de date ce contin mai
multe segmentari am impartit setul de date in matrici separate ce contin fiecare segmentare si am aplicat algoritmul pe fiecare dintre ele iar rezultatele le-am adaugat intr-un singur fisier.
