# INZYNIERKA



## Co zrobic

- Zacząć trzeba od udanej detekcji komórek na zdjęciu , najpierw z OpenCv I potem się zobaczy czy jest to dość  “MOJE” zeby mogło zostać ale pewnie trzeba będzie po tym zrobić moje własne główne funkcje
- Klasyfikacja komórek , jak to ostatecznie zrobić , czy jakis machine learning czy nie ( niby może byc ale na pewno nie będzie to sieć neuronowa bo to nie ma sensu)
- Prezentacja/GUI  albo program na kompa(łatwo)  ,  albo  aplikacja na telefon(raczej trudniejsze)   ,   albo strona internetowa(nie wiadomo) // coś że wysyła sie komuj link I ktoś moze sobie otworzyć w przeglądarce I tam jest apka

## Co można dorobić

- Można jakoś spróbować ogarnąć lepsze wybieranie konturów :
      Pętla po wszystkich pikselach działa i zamienia co ciemniejsze piksele na czarne zeby threshold lepiej zadziałał ale trwa to ok. 15 min.
      ale nawet ta metoda nie koniecznie działa lepiej niż adaptiveThreshold


## Istotne rzeczy 
- najlepiej trzymać się wersji OpenCv  4.5.5.62  ,  dla tej wersji działa dokumentacja w pycharmie Profesional na Linux
- Machine Learning tutaj raczej cieżko z kminieniem Sieci Neuronowych ale możnaby użyć do klasyfikacji inaczej ML
      Możnaby wykorzystać uczenie maszynowe bez sieci neuronowych , cos w tą strone powinno być w OpenCV
      Ewentualnie ciekawsza rzecz to żeby jednak użyć tych sieci neuronowych ale do rozpoznawania komórek  , podzielić zdjęcia na pojedyncze komórki i nauczyć sieć rozpoznawania czy jest 
      jasna czy ciemna , ALEEE nie wiadomo czy jest tu sens się wgl w to bawić  CHOCIAZZZ  mogłoby być fajne

## Przydatne tak jakby co 
- python3 -m pip install --force-reinstall --no-cache -U opencv-python==4.5.5.62   -    to w terminalu w pycharm zeby przeinstalować opencv
- pip3 install Shift+CShift+Copencv-python





## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.com/my_projects76/inzynierka.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.com/my_projects76/inzynierka/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Automatically merge when pipeline succeeds](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

