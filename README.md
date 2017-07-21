## Bugzilla Tag Recommendation Engine

#### Motivation
We get many questions or issue reports from live customers regarding our products. And this is current process pipeline for handling them.
1. Customer post an `inquiry`
2. Support identifies the `issue type`
3. Support creates a `bugzilla ticket`

Given limited resources, our goal is to optimize the process to serve as many tickets as possible without compromising response SLA.

This program attempts to semi-automate the `second step` by identifying the issue based on past data.

#### Description
This a model based Bugzilla tag recommendation engine. Engine is powered by set of classifiers trained on 4k bugzilla tickets in AppVolumes product. Here is the sample screenshot of UI which predicts bugzilla tags given a new user inquiry.

![WebUI](https://onevmw-my.sharepoint.com/personal/juhyeongy_vmware_com/_layouts/15/guestaccess.aspx?docid=0b240543cb97c49f9afc4291d7a555884&authkey=AZYwQMlFC371cA8saLEoYgE&expiration=2017-07-31T06%3a33%3a54.000Z)

#### Architecture
![Architecture](https://onevmw-my.sharepoint.com/personal/juhyeongy_vmware_com/_layouts/15/guestaccess.aspx?docid=01d0677c77a224f13bcb57c2a509fe253&authkey=AQa2pHHZCPMZj-M70_PmlKQ&expiration=2017-07-31T09%3a36%3a21.000Z)

#### Models
Models are `convolutional neural network` with a `word embedding layer` custom trained with bugzilla corpus. Each model is individually trained with ~4k bugzilla tickets, and supervise learning is used against training data set of `(bugzilla-ticket-contents, labels: category, priority, severity, component, type)`

Here is dataset sample
![DataSample](https://onevmw-my.sharepoint.com/personal/juhyeongy_vmware_com/_layouts/15/guestaccess.aspx?docid=021cebb8598ab43fd9c5f8c02db61073e&authkey=AeJ7G7-C8rfWSJujn25NLkM&expiration=2017-07-31T10%3a26%3a35.000Z)

Here is train/test validation report sample. Validation accuracy scores are bounded between 40~60%
```
Training Category-Classifier Now...
Train on 3288 samples, validate on 823 samples
Epoch 10/20
3288/3288 [==============================] - 91s - loss: 1.3141 - acc: 0.5289 - val_loss: 1.3562 - val_acc: 0.5431
Training Component-Classifier Now...
Train on 3288 samples, validate on 823 samples
Epoch 1/20
3288/3288 [==============================] - 101s - loss: 3.1712 - acc: 0.2205 - val_loss: 2.9410 - val_acc: 0.1324
```

#### Samples
AppCapture related question from a community discussion. This is identified as `(AppCapture, General, Serious, Defect)`

```
1. Re: App Stacks and OS Version
 techguy129
Enthusiast
techguy129 Jul 20, 2017 1:30 PM (in response to RanjnaAggarwal)
They are not OS version specific. I create appstacks on windows 10 and use them on server 2012 and server 2016 with no issues. Be sure to test that the application in the appstack do actually work against the different OS's.

When creating an appstack, it will be only available for that version. You have to edit the appstack in the web manager after you provision it to specify which OS's the appstack is allowed to mount too.
```

Agent related question from a community discussion. This is identified as `(Product, Agent, Serious, Defect)`

```
We have packaged MS Office Pro Plus 2013 in App Volumes 2.10.  We have two issues, slowness and configuration boxes appearing.

1. When the user adds an ActiveX control within MS Access they click ‘Create’ ‘Form Design’ then select ‘ActiveX Controls’ by clicking the drop down menu in ‘Form Design Tools’.  This action in itself takes roughly 1.5 minutes.  I think at this point maybe the root classes in the registry are being accessed because if I open ‘regedit’ and click on ‘HKEY_Classes_Root’ it takes a long time before it opens.

2. When the user tries to add an ActiveX Control to the form a Windows Installer box appears saying Preparing to Install, click cancel and an MS Office Pro Plus box appears saying ‘Please wait while Windows Configures MS Office Pro Plus 2013 – gathering required information’ the user has to click cancel again.  The ActiveX component is then added to the form.  This happens every time the user views a database that holds this particular ActiveX control, all other controls work okay.

I mirrored this scenario on a Windows 7 VM and it works fine, as soon as I package it with App Volumes and attach it to a user we get all the issues.

Our Golden image is a Windows7 64bit VM and our App Vol VM is a clone of this, our desktop VMs are non-persistent.  We have no other MS Office applications on our base image.

We have this set in the registry HKCU\Software\Microsoft\Office\15.0\Access\Options /v NoReReg /t REG_DWORD /d 1

The error also doesn’t happen in MS Office 2010 App Stack. Office 2010 and 2013 have been installed the same way.

The configuring issue is driving me mad as it randomly happens in other applications.

Any advice will be very much appreciated; thanks Anita R.
```

### Repo Structure
- classifiers -> models + training/update
- crawler -> bugzilla tickets crawler
- data-service -> prediction api service
- web -> Web UI app
