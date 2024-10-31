# ---------------------------------------
# Step 1: Clear dist folder and create CaptureDisplayX77.app
# ---------------------------------------
#Use the below command to create the spec file, once that is ready use spec file with pyinstaller
#when onefile argument is given - extraction of app happens at every launch - which makes the launch slow, so keep that in mind, this simplicity of app file structure has a cost
#pyinstaller --noconsole --name CaptureDisplayX77 --icon=./assets/CaptureDisplayX77.icns /Users/sachinjeph/Desktop/CaptureDisplay/main.py
#But for some reason creating the app using the below flag allows notarisation else notarisation does not work
#pyinstaller --onefile --windowed ./main.py --name "CaptureDisplayX77" --icon=./assets/CaptureDisplayX77.icns
rm -r dist
rm -r build
#create a bundle identifer from appstoreconnect and use that in the spec file
pyinstaller CaptureDisplayX.spec
cd dist
find CaptureDisplayX77.app -name .DS_Store -delete
cd ..
#If app is crashing on launch- try running the terminal version in the dist folder it will give you logs and you can see what's going wrong
# ---------------------------------------
# Step 2: Signing the CaptureDisplayX77.app
# ---------------------------------------
#Signing needs to happen with a certificate- if you use developer id certificate like the one here the notarization does not work, we need to use self signed certificate like 'CaptureDisplayX77'. Example developer application id certificate -> EC24DE91843FE9267B360FA70CAFAF873E92AC72 "Developer ID Application: sachin jeph (34XRA32US9)"
#Use this command to show the list of available certificates -> security find-identity -p basic -v
#-s hash, this hash is corresponding to the developer id application certificate, you see you current certificates and hashes using the command $security find-identity -p basic -v. We need a developer id application certificate to notarize the apps
#signing the .app where -s denotes teh signture certificate we need to provide. Please use self signed certificate otherwise problems occur in notarization wher notarization fails
#You can create a self signed certificate from Keychain access. Make sure to open the certiciate -> Trust section -> Set Code Signing as 'Always Trust'
#Or okay i just realised you can use your developer certificate and open the certiciate -> Trust section -> Set Code Signing as 'Always Trust' as well, and that should work as well for code signing certificates
#I am just thinkning the code signing using developer certificate might be better than self signed certicate no? i am not entirely sure about this for now, but for now using developer certiciate for code signing
# codesign --deep --force --verbose -s "CaptureDisplayX77" "./dist/CaptureDisplayX77.app"
#we must sign with Developer ID Application certificate in order to notarize the app
#verify codesign -> codesign -vvv --deep --strict ./dist/CaptureDisplayX77.app
codesign -s "EC24DE91843FE9267B360FA70CAFAF873E92AC72" -v --deep --force --timestamp --entitlements entitlements.plist -o runtime "./dist/CaptureDisplayX77.app"

# ---------------------------------------
# Step 3: Convert the application bundle to a DMG (macOS disk image)
# ---------------------------------------
echo "Creating DMG installer..."
sleep 5

#Visit https://github.com/create-dmg/create-dmg for more information on create-dmg
# Create the DMG
# Ensure you have 'create-dmg' installed. If not, install using 'brew install create-dmg'
create-dmg  --volname "CaptureDisplayX77" --volicon "/Users/sachinjeph/Desktop/CaptureDisplay/assets/CaptureDisplayX.ico" --icon-size 100 --app-drop-link 425 120 "./dist/CaptureDisplayX77.dmg" "./dist/CaptureDisplayX77.app/"
# ---------------------------------------
# Step 4: Signing the CaptureDisplayX77.dmg
# ---------------------------------------
#Signing the dmg
echo "Signing DMG..."
codesign -s "EC24DE91843FE9267B360FA70CAFAF873E92AC72" -v --deep --force --timestamp --entitlements entitlements.plist -o runtime "./dist/CaptureDisplayX77.dmg"

echo "Packaging and signing complete. You can find the DMG installer in the dist/ directory."

# ---------------------------------------
# Step 5: Notarising the DMG (macOS disk image)
# ---------------------------------------
#For performing notarization you will need a keychain profile, to create it you can run the below command and then on screen steps
#xcrun notarytool store-credentials
#Here you will be asked to create an app specific password which you can generate from account.appleid.com, its simple
#Once your keychain profile is ready it will be there in keychain
#NOT NOTARIZING CAPTUREDISPALY BECAUSE I WAS ONLY ABLE TO NOTARIZE THE APP IN ONEFILE MODE WHICH IS SLOW ON EVERY LAUNCH DUE TO DECODING OF FILES ON LAUNCH
# echo "Notarising DMG..."
# #Notarisation step- takes some time so thats why they are commented
# xcrun notarytool submit "./dist/CaptureDisplayX77.dmg" --keychain-profile "CaptureDisplayX77" --wait
# #Staple the dmg, this allows for app verification without internet access on user's side
# xcrun stapler staple "./dist/CaptureDisplayX77.dmg"
#Validate stapler step
#xcrun stapler validate "./dist/CaptureDisplayX77.dmg"
#To verify if a dmg is notarised or not
# spctl -a -vvv -t install "./dist/CaptureDisplayX77.dmg"
#To see history of your notarization request run the below command
# xcrun notarytool history --keychain-profile "CaptureDisplayX77"
#To get the detailed log of notarization step use the below commnad
#xcrun notarytool log "./dist/CaptureDisplayX77.dmg" --keychain-profile "CaptureDisplayX77"
#if notarization fails you can see notary logs which has the reasons for failure using the command where this id is the submission id which you would see when you run the initial notarization command
#xcrun notarytool log 8e03c9e3-219f-48be-a22d-4ba4846490c3 --keychain-profile "CaptureDisplayX77"

#To see the list of certificates
#security find-identity -p basic -v
#Maybe write some upload script here to upload dmg for customers